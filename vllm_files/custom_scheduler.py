import asyncio
import time
import urllib.parse
from typing import Callable, Any, Awaitable, Dict, Optional, Set
import aiohttp
from vllm.logger import init_logger

logger = init_logger(__name__)



class ModelScheduler:
    """
    Serialize requests across models so only one model is awake/generating at a time.
    Maintains a global FIFO queue. Worker processes requests in queue order.
    """

    def __init__(
        self,
        base_url: str,
        model_to_port_map: Dict[str, int],
        session: aiohttp.ClientSession,
        sleep_level: int = 1,
        check_every: float = 0.1,
        timeout: float = 10
    ):
        self.base_url = base_url.rstrip("/")
        self.model_to_port_map = dict(model_to_port_map)
        self.session = session
        self.sleep_level = sleep_level
        self.check_every = check_every
        self.timeout = timeout

        logger.info(f"Creating ModelScheduler with baseurl {self.base_url}")

        # FIFO queue of (model_name, coro_factory, future)
        # coro_factory: Callable[[str], Awaitable[Any]] which will be called with the model-specific base_url
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = asyncio.create_task(self._worker_loop())

        # currently awake model name, or None if none awake
        self._active_model: Optional[str] = None
        # lock to protect switching (not strictly necessary with worker loop single-threaded, but safe)
        self._switch_lock = asyncio.Lock()

        # track inflight tasks per model
        self._inflight_by_model: Dict[str, Set[asyncio.Task]] = {}

        self._closed = False
        
    async def set_active_model(self):
        # Check if any model is awake, then set _active_model
        for model in self.model_to_port_map.keys():
            tmp_base_url = self._model_base_url(model)
            clean_base_url = self._strip_path(tmp_base_url)
            if not await self._is_server_sleeping(clean_base_url):
                logger.info(f"Found awake model {model}. Setting it as active model")
                self._active_model = model
                # Break, since there can only be one active model currently (one GPU)
                break

    def _model_base_url(self, model_name: str) -> str:
        """Return base_url but with the port replaced by model_to_port_map[model_name]."""
        
        if model_name not in self.model_to_port_map:
            raise KeyError(f"Unknown model: {model_name}")

        parsed = urllib.parse.urlparse(self.base_url)
        # build new netloc with replaced port
        host = parsed.hostname
        port = self.model_to_port_map[model_name]
        # preserve username/password if present
        netloc = ""
        if parsed.username:
            netloc += parsed.username
            if parsed.password:
                netloc += f":{parsed.password}"
            netloc += "@"
        netloc += f"{host}:{port}"

        new_parsed = parsed._replace(netloc=netloc)
        return urllib.parse.urlunparse(new_parsed)

    def _strip_path(self, url: str) -> str:
        """
        Removes any path portion from a URL (everything after host:port).
        Returns only scheme://host:port
        """
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc
        return f"{parsed.scheme}://{netloc}"

    async def schedule(self, coro_factory: Callable[[str], Awaitable[Any]], model_name: str) -> Any:
        """
        Enqueue a request and wait for its result.

        coro_factory is a callable that receives the model-specific base_url (string)
        and returns an awaitable that actually performs the request and returns the result.
        """
        if self._closed:
            raise RuntimeError("Scheduler closed")

        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((model_name, coro_factory, fut))
        return await fut

    async def _wait_for_all_inflight(self):
        """Await all currently inflight tasks (across all models)."""
        # Snapshot tasks
        tasks = []
        for s in self._inflight_by_model.values():
            tasks.extend([t for t in s if not t.done()])
        if not tasks:
            return
        # Wait for them to finish. We return exceptions to callbacks handling futures.
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _switch_to_model(self, target_model: Optional[str]):
        """
        Switch active model from current to target_model:
        - If target_model is None -> sleep current active model (if any) and set active=None.
        - If target_model == active -> do nothing.
        - If active != target_model:
            - If there is an active model, put it to sleep (_action_server "sleep")
            - Wake up target_model (_action_server "wakeup")
        This function will block until actions are complete via _action_server.
        
        Note: If switching between different non-None models, wait for inflight requests to finish
        before performing the sleep/wakeup actions.
        """
        logger.info(f"Switching from {self._active_model} to {target_model}")
        async with self._switch_lock:
            if target_model == self._active_model:
                return

            # If there are inflight tasks, wait for them to finish before switching model
            if self._inflight_by_model:
                await self._wait_for_all_inflight()

            # Sleep current active if exists
            if self._active_model is not None:
                tmp_base_url = self._model_base_url(self._active_model)
                clean_base_url = self._strip_path(tmp_base_url)
                try:
                    await self._action_server(
                        base_url=clean_base_url,
                        action="sleep"
                    )
                except Exception as e:
                    # propagate or just log; we choose to raise so caller sees failure
                    raise RuntimeError(f"Failed to sleep model {self._active_model}: {e}") from e
                self._active_model = None
                
            # Wake target if not None
            if target_model is not None:
                tmp_base_url = self._model_base_url(target_model)
                clean_base_url = self._strip_path(tmp_base_url)
                try:
                    await self._action_server(
                        base_url=clean_base_url,
                        action="wakeup",
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to wake model {target_model}: {e}") from e
                self._active_model = target_model

    async def _worker_loop(self):
        """
        Continuously process queue items in FIFO order.
        For each request:
          - ensure target model is awake (sleep previous active if switching)
          - call the coro_factory(base_url_for_model) and and don't await so same-model requests run concurrently
          - set the future with the result/exception
          - peek at next queue item: if same model, keep it awake; otherwise, sleep the current model
            (sleeping is deferred to the switch when next item is processed to avoid extra sleeps).
          - When switching models, ensure we wait for inflight tasks to complete (handled in _switch_to_model).
        """
        try:
            while True:
                model_name, coro_factory, fut = await self._queue.get()
                logger.info("New item for worker loop has arrived")

                # If scheduler is closing, fail requests
                if self._closed:
                    if not fut.done():
                        fut.set_exception(RuntimeError("Scheduler closed"))
                    self._queue.task_done()
                    continue

                try:
                    # ensure target model is awake (this will wait for inflight tasks if switching)
                    if self._active_model != model_name:
                        await self._switch_to_model(model_name)

                    logger.info(f"Model {model_name} is awake now!")

                    # Prepare the awaitable from coro_factory.
                    awaitable = coro_factory()

                    # Create task, track it, and do NOT await it here (non-blocking dispatch)
                    task = asyncio.create_task(awaitable)

                    # ensure inflight set exists
                    s = self._inflight_by_model.setdefault(model_name, set())
                    s.add(task)

                    # define callback to deliver result to the waiting future and cleanup bookkeeping
                    def _done_callback(t: asyncio.Task, _fut=fut, _m=model_name):
                        try:
                            res = t.result()
                        except Exception as e:
                            if not _fut.done():
                                _fut.set_exception(e)
                        else:
                            if not _fut.done():
                                _fut.set_result(res)
                        # remove from inflight set
                        try:
                            ss = self._inflight_by_model.get(_m)
                            if ss and t in ss:
                                ss.remove(t)
                                if not ss:
                                    # remove empty entry
                                    del self._inflight_by_model[_m]
                        except Exception:
                            pass

                    task.add_done_callback(_done_callback)

                    # We dispatched the request — mark the queue item as done (we consider dispatch complete)
                    self._queue.task_done()
                    logger.info("Task dispatched.")

                except Exception as e:
                    if not fut.done():
                        fut.set_exception(e)
                    # ensure queue task count updated in error case
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass

        except asyncio.CancelledError:
            # worker cancelled: cleanup
            pass

    async def close(self):
        """Close the scheduler: stop accepting new, wait for remaining work, sleep active model and stop worker."""
        logger.info("Closing the scheduler")
        self._closed = True

        # Wait for queue to be processed (all items dispatched) then wait for inflight tasks to finish.
        # Note: worker marks queue.task_done() when it dispatches items, so queue.join() waits until dispatched.
        try:
            await self._queue.join()
        except Exception:
            # If join fails for any reason, continue to inflight wait
            pass

        # Wait for any tasks currently running
        if self._inflight_by_model:
            await self._wait_for_all_inflight()

        # Sleep active model if any
        if self._active_model is not None:
            try:
                await self._switch_to_model(None)
            except Exception:
                pass

        # Finally cancel worker task
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    # Helper functions for sleep/wakeup management (unchanged)
    async def _post(self, url: str, data=None):
        try:
            if data is None:
                async with self.session.post(url) as r:
                    await r.read()
                    return r.status
            else:
                async with self.session.post(url, json=data) as r:
                    await r.read()
                    return r.status
        except Exception as e:
            raise ValueError(f"Error sending POST to {url}: {e}")

    async def _wakeup(self, base_url: str):
        """Execute wake-up behavior based on sleep level."""
        
        logger.info(f"Executing wakeup for level {self.sleep_level}")
        
        # Wake up
        await self._post(f"{base_url}/wake_up")
        
        if self.sleep_level == 2:
            # Reload weights
            await self._post(f"{base_url}/collective_rpc", {"method": "reload_weights"})
            # Reset prefix cache
            await self._post(f"{base_url}/reset_prefix_cache")

    async def _sleep(self, base_url: str):
        """Execute sleep behavior based on sleep level."""
        
        logger.info(f"Executing sleep level {self.sleep_level}")
        await self._post(f"{base_url}/sleep?level={self.sleep_level}")

    async def _is_server_sleeping(self, url: str):
        """Check if the server is currently in sleep mode."""
        
        async with self.session.get(f"{url}/is_sleeping") as r:
            data = await r.json()
            return data.get("is_sleeping", False)

    async def _action_server(self, base_url: str, action: str):
        start = time.time()

        if action == "wakeup":
            await self._wakeup(base_url)
        elif action == "sleep":
            await self._sleep(base_url)
        else:
            raise ValueError(f"Unsupported action: {action}")

        logger.info(f"action {action} is sent to {base_url}. Waiting for it to complete!")
        # Wait until action is complete
        start_time = time.time()
        while True:
            async with self.session.get(f"{base_url}/is_sleeping") as r:
                if r.status == 200:
                    data = await r.json()
                    is_sleeping = data.get("is_sleeping", False)
                    if action == "sleep" and is_sleeping:
                        end_time = time.time()
                        logger.info(f"Server is sleeping after {end_time - start:.3f} seconds.")
                        break

                    if action == "wakeup" and not is_sleeping:
                        end_time = time.time()
                        logger.info(f"Server is wakeup after {end_time - start:.3f} seconds.")
                        break

            elapsed = time.time() - start_time
            if elapsed >= self.timeout:
                raise Exception(f"Timed out after waiting {self.timeout} seconds to take action {action}.")

            await asyncio.sleep(self.check_every)

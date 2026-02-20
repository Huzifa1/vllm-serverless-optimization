for bt in 4; do
    echo "Running test with max batched tokens = $bt"
    bash server.sh $bt &
    SERVER_PID=$!
    sleep 60

    # python client.py mix > logs/mix_$bt.log
    # python client.py prefill > logs/prefill_$bt.log
    python client.py decode > logs/decode_$bt.log

    kill $SERVER_PID
    pid=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    kill -9 $pid
    sleep 10
 done
for i in {0..2}
do
    # Delete old container if exists
    if [ "$(sudo docker ps -aq -f name=vllm-$i)" ]; then
        sudo docker stop vllm-$i
        sudo docker rm -f vllm-$i
    fi

    device=$(bash get-mig-uuid.sh $i)
    port=$((8000 + i))
    model="llama3-3b"
    if [ $i -eq 1 ]; then
        model="qwen-1.8b"
    elif [ $i -eq 2 ]; then
        model="qwen2.5-3b"
    fi

    sudo docker run -d --rm --gpus "\"device=$device\"" \
    -p $port:8000 \
    --name vllm-$i \
    -v /local/huzaifa/workspace/models:/local/huzaifa/workspace/models \
    vllm/vllm-openai:v0.10.1.1 \
    --model /local/huzaifa/workspace/models/$model

    # Stream logs live AND save to file (backgrounded)
    sudo docker logs -f vllm-$i 2>&1 > logs/vllm-$i.log &
done
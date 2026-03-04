export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

start_time=$(date +%s)
echo "start_time: ${start_time}"

nohup python -m torch.distributed.launch --nproc_per_node 8 --use-env main.py --mode train  > ./train.log 2>&1 & 

# nohup python main.py --mode train > ./train.log 2>&1 & 

# --master_port=16661
#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

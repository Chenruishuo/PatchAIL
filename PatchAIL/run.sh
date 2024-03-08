source activate imitation

for seed in 0 1 2
do     
        CUDA_VISIBLE_DEVICES=0 python train.py agent=encirl suite=atari obs_type=pixels suite/atari_task=pong algo_name=encairl num_demos=20 replay_buffer_size=1000000 agent.eta=0 seed=$seed &
        sleep 1
done
wait
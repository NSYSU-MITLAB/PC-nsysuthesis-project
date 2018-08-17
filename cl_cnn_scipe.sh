#python3 multi_task_mnfc.py --epoch=25 --load_model=False --learning_rate=0.001
#python3 multi_task_mnfc.py --epoch=25 --load_model=True --learning_rate=0.00001
#python3 multi_task_mnf.py --epoch=25 --load_model=False --learning_rate=0.001
#python3 multi_task_mnf.py --epoch=25 --load_model=True --learning_rate=0.00001
#python3 multi_task_mn.py --epoch=25 --load_model=False --learning_rate=0.001
#python3 multi_task_mn.py --epoch=25 --load_model=True --learning_rate=0.00001
#python3 multi_task_fn.py --epoch=25 --load_model=False --learning_rate=0.001
#python3 multi_task_fn.py --epoch=25 --load_model=True --learning_rate=0.00001
#python3 multi_task_mf.py --epoch=25 --load_model=False --learning_rate=0.001
#python3 multi_task_mf.py --epoch=25 --load_model=True --learning_rate=0.00001

#python3 single_task_cnn.py --epoch=25 --load_model=False --learning_rate=0.001 --multi_task=False --datasets=0
#python3 single_task_cnn.py --epoch=25 --load_model=True --learning_rate=0.00001 --multi_task=False --datasets=0

for i in 0
do
	for s in 28
	do
		for c in 0 1 2 3 4
		do
			python3 single_task_cnn.py --epoch=25 --load_model=False --learning_rate=0.001 --datasets=$i --imgsize=$s --model=$c
			python3 single_task_cnn.py --epoch=25 --load_model=True --learning_rate=0.00001 --datasets=$i --imgsize=$s --model=$c
		done
	done
done


#for i in 0 1 2
#do
#	for s in 28 21 14 7 5
#	do
#		for c in 0 1 2 3 4
#		do
#			python3 single_task_cnn.py --epoch=25 --load_model=False --learning_rate=0.001 --datasets=$i --imgsize=$s --model=$c
#			python3 single_task_cnn.py --epoch=25 --load_model=True --learning_rate=0.00001 --datasets=$i --imgsize=$s --model=$c
#		done
#	done
#done

#for n in 1 2 3 4 6
#do
	#for ((i=0;i<=2; i=i+1))
	#do
		#python3 single_task_cnn.py --epoch=25 --load_model=False --learning_rate=0.001 --multi_task=False --datasets=$i --d_component=$n
		#python3 single_task_cnn.py --epoch=25 --load_model=True --learning_rate=0.00001 --multi_task=False --datasets=$i --d_component=$n
	#done
#done


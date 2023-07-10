import matplotlib.pyplot as plt
# filename = 'checkpoints/P_Net_draw_picture_init.txt'
# filename = 'checkpoints/R_Net_draw_picture_init.txt'
filename = 'checkpoints/O_Net_draw_picture_init.txt'
x=[]
y=[]
x_label="epoch"
y_label="loss"
title=" Line Chart of Loss and Iteration Number for O-Net"
with open(filename, 'r',encoding='utf8') as f:
	for line in f:
		epoch, loss, cls_loss, bbox_offset_loss,landmark_offset_cls = line.strip().split()
		epoch = int(epoch)  # 去掉epoch后面的冒号
		loss = float(loss)
		x.append(epoch)
		y.append(loss)
plt.plot(x,y,marker="*")
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.show()

import numpy as np

from SETR_1D_config import *
def norm_data_IMU(acc_data,gyro_data):
    acc_data=acc_data[acc_data.shape[0]-gyro_data.shape[0]:]
    return acc_data,gyro_data

#1、分类数据集
#检测index到index+spilt_len内是否仅包含一段完整的语义数据,diff_number为加速度和角速度数据量不同带来的偏移,
def have_complete_semantics(index,line,diff_number):
    semantics_num=int(len(line)/2)
    if semantics_num!=5:
        print("标签错误")
    for i in range(semantics_num):
        if i==0:
            if index + diff_number <= line[2 * i] * sample_frequent and line[2 * i + 1] * sample_frequent<= (index + spilt_len + diff_number) <= line[2 * (i+1)] * sample_frequent:
                return 1
        elif i==semantics_num-1:
            if line[(2 * i) - 1] * sample_frequent <= index + diff_number <= line[2 * i] * sample_frequent and line[2 * i + 1] * sample_frequent <= (index + spilt_len + diff_number):
                return 1
        else:
            if line[(2 * i)-1] * sample_frequent<=index + diff_number <= line[2 * i] * sample_frequent and line[2 * i + 1] * sample_frequent<= (index + spilt_len + diff_number) <= line[2 * (i+1)] * sample_frequent:
                return 1
    return 0
def get_spilt_data():
    filepath=r"D:\my_data\cam\original"
    sample_number=10
    X_all=np.zeros((0,6,spilt_len))
    y_all = np.zeros((0,1))
    for i in range(gesture_number):
        #提取分割文件
        txt_path=os.path.join(filepath, str(i),"segment.txt")
        # 使用with open打开文件，确保文件操作完成后会自动关闭文件
        with open(txt_path, 'r') as file:
            # 使用readlines()方法读取文件中的所有行，每行作为一个字符串元素存储在列表lines中
            lines = file.readlines()
        for j in range(sample_number):
            #提取整段IMU数据
            filepath_temp = os.path.join(filepath,str(i),str(j)+ ".xls")
            acc_data = (pd.read_excel(filepath_temp, sheet_name="Accelerometer")).values[:, 1:4]  # *************************
            gyro_data = (pd.read_excel(filepath_temp, sheet_name="Gyroscope")).values[:, 1:4]  # *************************
            diff_number=acc_data.shape[0]-gyro_data.shape[0]
            acc_data, gyro_data = norm_data_IMU(acc_data, gyro_data)
            IMU_data = np.hstack((acc_data, gyro_data))
            #提取分割数据
            line=(lines[j].strip()).split(",")
            line = [eval(word) for word in line]
            index=0
            while index+spilt_len<IMU_data.shape[0]:
                if have_complete_semantics(index,line,diff_number):
                    new_temp = np.expand_dims(IMU_data[index:index + spilt_len].T, axis=0)
                    X_all = np.concatenate((X_all, new_temp), axis=0)
                    y_all = np.vstack((y_all, i))
                index+=random.randint(0, step_len)
    save_path = r"D:\my_data\cam\classification\train"
    save_path_X = os.path.join(save_path, "X_all.npy")
    save_path_y = os.path.join(save_path, "y_all.npy")

    X_all = X_all.astype(np.float32)
    y_all = (y_all.squeeze()).astype(np.longlong)
    # 存数据
    np.save(save_path_X, X_all)
    np.save(save_path_y, y_all)
    # 读数据
    X_all = np.load(save_path_X)
    y_all = np.load(save_path_y)
    return X_all, y_all
def get_test_data(save_path,semantic_index,number_index=0):
    filepath = os.path.join(save_path, semantic_index,str(number_index)+".xls")
    acc_data = (pd.read_excel(filepath, sheet_name="Accelerometer")).values[:, 1:4]  # *************************
    gyro_data = (pd.read_excel(filepath, sheet_name="Gyroscope")).values[:, 1:4]  # *************************
    acc_data, gyro_data = norm_data_IMU(acc_data, gyro_data)
    IMU_data = np.hstack((acc_data, gyro_data))
    IMU_data=IMU_data.T
    return IMU_data
##2、分割数据集（双语义）
def get_mask_binary(mask_len,line,diff_number):
    mask=np.zeros((mask_len))
    segmentation_array=(np.array(line)*sample_frequent-diff_number).astype(int)
    for i in range(int(len(line)/2)):
        mask[segmentation_array[2*i]:segmentation_array[2*i+1]]=1
    return mask
def get_segmentation_binary(semantic_index=0):
    # filepath = r"D:\my_data\cam\original"
    # sample_number = 10
    # X_all = np.zeros((0, 6, spilt_len))
    # mask_all = np.zeros((0, spilt_len)).astype(int)
    #
    # txt_path = os.path.join(filepath, str(semantic_index), "segment.txt")
    # # 使用with open打开文件，确保文件操作完成后会自动关闭文件
    # with open(txt_path, 'r') as file:
    #     # 使用readlines()方法读取文件中的所有行，每行作为一个字符串元素存储在列表lines中
    #     lines = file.readlines()
    # for j in range(sample_number):
    #     # 提取整段IMU数据
    #     filepath_temp = os.path.join(filepath, str(semantic_index), str(j) + ".xls")
    #     acc_data = (pd.read_excel(filepath_temp, sheet_name="Accelerometer")).values[:, 1:4]  # *************************
    #     gyro_data = (pd.read_excel(filepath_temp, sheet_name="Gyroscope")).values[:, 1:4]  # *************************
    #     diff_number = acc_data.shape[0] - gyro_data.shape[0]
    #     acc_data, gyro_data = norm_data_IMU(acc_data, gyro_data)
    #     IMU_data = np.hstack((acc_data, gyro_data))
    #     # 提取分割数据,得到mask
    #     line = (lines[j].strip()).split(",")
    #     line = [eval(word) for word in line]
    #     mask=get_mask_binary(len(IMU_data),line, diff_number)
    #
    #     index = 0
    #     while index + spilt_len < IMU_data.shape[0]:
    #         X_temp = np.expand_dims(IMU_data[index:index + spilt_len].T, axis=0)
    #         mask_temp=np.expand_dims(mask[index:index + spilt_len], axis=0)
    #         X_all = np.concatenate((X_all, X_temp), axis=0)
    #         mask_all = np.concatenate((mask_all,mask_temp), axis=0)
    #         index += random.randint(0, step_len)
    #         print(j,index)
    save_path = r"D:\my_data\cam\segmentation\train"
    save_path_X = os.path.join(save_path, "X_all.npy")
    save_path_mask = os.path.join(save_path, "mask_all.npy")

    # X_all = X_all.astype(np.float32)
    # mask_all = mask_all.astype(np.float32)
    # 存数据
    # np.save(save_path_X, X_all)
    # np.save(save_path_mask, mask_all)
    # 读数据
    X_all = np.load(save_path_X)
    mask_all = np.load(save_path_mask)
    return X_all, mask_all
##3、分割数据集（多语义）
#注意此时也分类不同，静止也作为一个语义，放在mask的最后一个通道，如如果3语义时，左右移动为0，上下为1，静止为2
def get_mask_multi(gesture_index,mask_len,line,diff_number):
    mask=np.zeros((gesture_number,mask_len))
    segmentation_array=(np.array(line)*sample_frequent-diff_number).astype(int)
    for i in range(int(len(line)/2)):
        if i==0:
            mask[gesture_number-1, 0:segmentation_array[2 * i]] = 1
        else:
            mask[gesture_number - 1, segmentation_array[2*(i-1)+1]:segmentation_array[2 * i]] = 1
        mask[gesture_index,segmentation_array[2*i]:segmentation_array[2*i+1]]=1
    mask[gesture_number-1,segmentation_array[-1]:]=1
    return mask
def get_segmentation_multi():
    #存取数据
    save_path = r"D:\my_data\cam\segmentation_multi\train"
    save_path_X = os.path.join(save_path, "X_all.npy")
    save_path_mask = os.path.join(save_path, "mask_all.npy")

    #生成数据
    # filepath = r"D:\my_data\cam\original"
    # sample_number = 10
    # X_all = np.zeros((0, 6, spilt_len))
    # mask_all = np.zeros((0, gesture_number,spilt_len)).astype(int)
    # for i in range(gesture_number-1):
    #     txt_path = os.path.join(filepath, str(i), "segment.txt")
    #     # 使用with open打开文件，确保文件操作完成后会自动关闭文件
    #     with open(txt_path, 'r') as file:
    #         # 使用readlines()方法读取文件中的所有行，每行作为一个字符串元素存储在列表lines中
    #         lines = file.readlines()
    #     for j in range(sample_number):
    #         # 提取整段IMU数据
    #         filepath_temp = os.path.join(filepath, str(i), str(j) + ".xls")
    #         acc_data = (pd.read_excel(filepath_temp, sheet_name="Accelerometer")).values[:, 1:4]  # *************************
    #         gyro_data = (pd.read_excel(filepath_temp, sheet_name="Gyroscope")).values[:, 1:4]  # *************************
    #         diff_number = acc_data.shape[0] - gyro_data.shape[0]
    #         acc_data, gyro_data = norm_data_IMU(acc_data, gyro_data)
    #         IMU_data = np.hstack((acc_data, gyro_data))
    #         # 提取分割数据,得到mask
    #         line = (lines[j].strip()).split(",")
    #         line = [eval(word) for word in line]
    #         mask=get_mask_multi(i,len(IMU_data),line, diff_number)
    #         index = 0
    #         while index + spilt_len < IMU_data.shape[0]:
    #             X_temp = np.expand_dims(IMU_data[index:index + spilt_len].T, axis=0)
    #             mask_temp=np.expand_dims(mask[:,index:index + spilt_len], axis=0)
    #             X_all = np.concatenate((X_all, X_temp), axis=0)
    #             mask_all = np.concatenate((mask_all,mask_temp), axis=0)
    #             index += random.randint(0, step_len)
    #             print(j,index)
    # # 存数据
    # X_all = X_all.astype(np.float32)
    # mask_all = mask_all.astype(np.float32)
    # np.save(save_path_X, X_all)
    # np.save(save_path_mask, mask_all)
    #

    # 读数据
    X_all = np.load(save_path_X)
    mask_all = np.load(save_path_mask)
    return X_all, mask_all
#4、用来评估多语义分割模型
#三维连续分割结果变成离散的
def segmentation_result_discretization(segmentation_result):
    segmentation_result = segmentation_result.cpu().detach().numpy() if isinstance(segmentation_result, torch.Tensor) else segmentation_result
    discretization_result=np.zeros_like(segmentation_result)
    for i in range(segmentation_result.shape[0]):
        for k in range(segmentation_result.shape[2]):
            discretization_result[i,np.argmax(segmentation_result[i,:,k]),k]=1
    return discretization_result
#多值离散分割结果变成二元离散
def multi_to_binary(multi_mask):
    if len(multi_mask.shape)==3:
        multi_mask=multi_mask[0]
    binary_mask=np.zeros((multi_mask.shape[-1]))
    #第一个开始为1,1和-1交换
    first_temp=multi_mask[:,0]
    first_flag=-1
    binary_mask[0] = first_flag
    for i in range(1,len(binary_mask)):
        if np.dot(first_temp,multi_mask[:,i]):
            binary_mask[i]=first_flag
        else:
            first_flag*=-1
            binary_mask[i]=first_flag
        first_temp=multi_mask[:,i]
    binary_mask[binary_mask==-1]=0
    return binary_mask
def get_acc_segmentation_multi(discretization_result,target):
    target =target.cpu().detach().numpy() if target.is_cuda else target
    acc_avg=0
    for i in range(discretization_result.shape[0]):
        acc_temp=0
        for k in range(discretization_result.shape[2]):
            acc_temp+=np.inner(discretization_result[i,:,k],target[i,:,k])
        acc_avg+=acc_temp/discretization_result.shape[2]
    return acc_avg/discretization_result.shape[0]
#5、为多语义分割模型合成数据，将2种语义片段粘贴一起
#按照类别生成
def splice_multi(splice_number=100):
    # 存取数据
    save_path = r"D:\my_data\cam\segmentation_multi\train_splice"
    save_path_X_splice = os.path.join(save_path, "X_splice.npy")
    save_path_mask_splice = os.path.join(save_path, "mask_splice.npy")
    #生成
    # X_splice = np.zeros((0, 6, spilt_len*2))
    # mask_splice = np.zeros((0, gesture_number,spilt_len*2)).astype(int)
    # X_all, mask_all=get_segmentation_multi()
    # splice_number=X_all.shape[0]
    # for i in range(3*splice_number):
    #     random_temp=np.random.randint(0,splice_number,(2))
    #     X_temp = np.expand_dims(np.hstack((X_all[random_temp[0]],X_all[random_temp[1]])), axis=0)
    #     mask_temp=np.expand_dims(np.hstack((mask_all[random_temp[0]],mask_all[random_temp[1]])), axis=0)
    #     X_splice = np.concatenate((X_splice, X_temp), axis=0)
    #     mask_splice = np.concatenate((mask_splice,mask_temp), axis=0)
    #     print(i)
    # # 存数据
    # X_splice= X_splice.astype(np.float32)
    # mask_splice = mask_splice.astype(np.float32)
    # np.save(save_path_X_splice, X_splice)
    # np.save(save_path_mask_splice, mask_splice)
    # 读数据
    X_splice = np.load(save_path_X_splice)
    mask_splice = np.load(save_path_mask_splice)
    return X_splice,mask_splice

if __name__=="__main__":
    X_splice,mask_splice =splice_multi()
    a=X_splice
    print(X_splice.shape)




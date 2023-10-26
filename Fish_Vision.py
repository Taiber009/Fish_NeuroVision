import numpy as np
import pandas as pd 
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import time
from datetime import datetime
import copy
from torchvision import transforms, models
import shutil 
from tqdm import tqdm


def show_input(input_tensor, title=''):
    plt.figure()
    image = input_tensor.permute(1, 2, 0).numpy()
    image = std * image + mean
    plt.imshow(image.clip(0, 1))
    plt.title(title)
    plt.show()
    plt.pause(0.001)

def show_plot(input_list_train,input_list_val, title=''):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.plot(input_list_train,label = 'train')
    plt.plot(input_list_val,label = 'val');
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()
    plt.pause(0.001)

def train_model(model, loss, optimizer, scheduler, num_epochs, train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch+1, num_epochs), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                if epoch>0:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            if phase=='train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(float(epoch_acc))
            else:
                val_loss_list.append(epoch_loss)
                val_acc_list.append(float(epoch_acc))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

    return model

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):# Расширение функции, добавляет путь
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

print("Запуск таймера")
start_time = time.time()

my_train_loss = []
my_train_acc = []
my_val_loss = []
my_val_acc = []

my_train_dir = 'my_train'
my_test_dir = 'my_test'
train_dir = 'temp_train'
test_dir = 'temp_test'
val_dir = 'temp_val'
class_names = []
my_epoch = 1# Эпохи
my_batch_size = 50# Кол-во картинок в батче

for root, dirs, files in os.walk(my_train_dir, topdown=False):
    for name in dirs:
        class_names.append(name)

csv_file = 'answer.csv'
if os.path.exists(csv_file):
    os.remove(csv_file)
if os.path.exists(train_dir):# Удаление временных файлов
    shutil.rmtree(train_dir)
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

for dir_name in [train_dir, val_dir]:
    for class_name in class_names:
        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

for class_name in class_names:# каждое 6 изображение уходит на валидационную выборку
    source_dir = os.path.join(my_train_dir, class_name)
    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
        if i % 6 != 0:
            dest_dir = os.path.join(train_dir, class_name) 
        else:
            dest_dir = os.path.join(val_dir, class_name)
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))

train_transforms = transforms.Compose([# Аугментация
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.Resize((336, 448)),
    #transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=my_batch_size, shuffle=True, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=my_batch_size, shuffle=False, num_workers=0)

print('Количество изображений в тренировочной выборке: '+str(len(train_dataset)))
print('Количество бачей: '+str(len(train_dataloader)))

X_batch, y_batch = next(iter(train_dataloader))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
#for x_item, y_item in zip(X_batch, y_batch):
#    show_input(x_item, title=class_names[y_item])# Посмотреть, какие картинка подались на вход


model = models.resnet18(pretrained=True)#resnet50, resnet152 и т.д.


model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Идет вычисление на '+str(device))
model = model.to(device)
loss = torch.nn.CrossEntropyLoss()# лосс-функция - кросс-энтропия
optimizer = torch.optim.Adam(model.parameters(),amsgrad=True, lr=1.0e-4)# Оптимайзер градиентного спуска

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

train_model(model, loss, optimizer, scheduler, my_epoch, my_train_loss, my_train_acc, my_val_loss, my_val_acc);# Обучение сетки

print("Прошло времени:"+str(round(time.time() - start_time,2))+" секунд")

unknown=os.path.join(test_dir, 'unknown')
if not os.path.exists(unknown):
    shutil.copytree(my_test_dir, os.path.join(test_dir, 'unknown'))
    
test_dataset = ImageFolderWithPaths(my_test_dir, val_transforms)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=my_batch_size, shuffle=False, num_workers=0)

model.eval()

show_plot(my_train_loss,my_val_loss,'loss')
show_plot(my_train_acc,my_val_acc,'accuracy')

test_predictions = []
test_img_paths = []
for inputs, labels, paths in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.set_grad_enabled(False):
        preds = model(inputs)
    test_predictions.append(
        torch.nn.functional.softmax(preds, dim=1)[:,:].data.cpu().numpy())
    test_img_paths.extend(paths)

test_img_paths = [p.lower().replace(my_test_dir+'\\'+'1'+'\\','').replace('.jpg','') for p in test_img_paths]

test_predictions = np.concatenate(test_predictions)

inputs, labels, paths = next(iter(test_dataloader))
paths = [p.lower().replace(my_test_dir+'\\'+'1'+'\\','').replace('.jpg','') for p in paths]

pred_class = []
for preds in test_predictions:
    max_index=0
    max = preds[max_index]
    for index in range(preds.size):
        if preds[index]>max:
            max=preds[index]
            max_index=index
    pred_class.append(class_names[max_index])


#for img, path, pred in zip(inputs, paths, pred_class):
#    show_input(img, title=path+' is '+pred)


answer_df = pd.DataFrame.from_dict({'id': test_img_paths, 'class': pred_class})
answer_df.set_index('id', inplace=True)
answer_df.to_csv(csv_file)# Тут все предсказания
# Реализовать сохранение сетки?
shutil.rmtree(train_dir)
shutil.rmtree(val_dir)
shutil.rmtree(test_dir)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GATConv 
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import classification_report
from sklearn.utils import resample
import random
from sklearn.model_selection import train_test_split 
# لایه تبدیل فرکانسی
class FrequencyTransformationLayer(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyTransformationLayer, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # تبدیل فرکانسی
        x_freq = torch.fft.fft2(x)
        x_freq = torch.abs(x_freq)  # فقط بخش‌های مغناطیسی فرکانس
        x_mod = torch.fft.ifft2(x_freq).real
        
        x = F.relu(self.conv1(x_mod))
        return x

# لایه جستجوی تکاملی
class EvolutionarySearchLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(EvolutionarySearchLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fc = nn.Linear(input_channels, output_channels)
        
    def forward(self, x):
        return self.fc(x)

# لایه Batch Normalization و Self Adjusting Layer
class SelfAdjustingFeatureLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAdjustingFeatureLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_matrix = nn.Parameter(torch.randn(in_channels, out_channels))  
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # تبدیل ویژگی‌ها با استفاده از ضرب ماتریسی
        x_transformed = torch.matmul(x, self.weight_matrix)  # عملیات ضرب ماتریسی
        x_transformed = self.bn(x_transformed)  # اعمال Batch Normalization
        return x_transformed

# لایه GAT
class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=num_heads, concat=True)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(out_channels * num_heads)
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        x = self.bn(x)
        return F.elu(x)
# مدل اصلی
class NovelConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes=3):
        super(NovelConvolutionalNetwork, self).__init__()
        
        # لایه‌های مدل
        self.freq_transform = FrequencyTransformationLayer(in_channels=3)
        self.gat_layer = GATLayer(in_channels=3 * 112 * 112, out_channels=600, num_heads=4)  # GAT layer
        self.evolutionary_search = EvolutionarySearchLayer(input_channels=600 * 4, output_channels=2048)  # Adjust input size
        self.drop1 = nn.Dropout(0.3)
        self.adjusting_layer = SelfAdjustingFeatureLayer(2048, 1024)
        self.drop2 = nn.Dropout(0.3)  
        self.fc1 = nn.Linear(1024, 256) 
        self.drop3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes) 

    def forward(self, data):
        # مطمئن شوید که ورودی از نوع Data است و دارای ویژگی x باشد
        x = data.x.view(-1, 3, 112, 112)  # تغییر شکل ورودی به ابعاد مناسب

        x = self.freq_transform(x)  # استخراج ویژگی‌های فرکانسی
        x = x.view(x.size(0), -1)  # flatten کردن خروجی برای لایه‌های بعدی
        
        # Create edge_index for GAT layer (this is a simple example, adjust as needed)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(x.device)  # Dummy edge index for example
        x = self.gat_layer(x, edge_index)  # Pass through GAT layer
        
        x = x.view(x.size(0), -1)  # Flatten output for subsequent layers
        x = self.evolutionary_search(x) 
        x = self.drop1(x)
        x = self.adjusting_layer(x)
        x = self.drop2(x)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)  
        x = self.fc2(x)  
        return x

def balance_dataset(dataset):
    benign = [data for data in dataset if data.y.item() == 0]
    malignant = [data for data in dataset if data.y.item() == 1]
    normal = [data for data in dataset if data.y.item() == 2]

    # Determine the number of samples for each class
    max_size = max(len(benign), len(malignant), len(normal))

    # Oversampling to balance classes
    benign_upsampled = resample(benign, replace=True, n_samples=max_size, random_state=123)
    malignant_upsampled = resample(malignant, replace=True, n_samples=max_size, random_state=123)
    normal_upsampled = resample(normal, replace=True, n_samples=max_size, random_state=123)

    balanced_dataset = benign_upsampled + malignant_upsampled + normal_upsampled
    return balanced_dataset

def load_new_data(save_folder='new_data'):
    data_list = []
    
    for filename in os.listdir(save_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(save_folder, filename)
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # Load corresponding label
            label_filename = filename.replace('.png', '.txt')
            label_path = os.path.join(save_folder, label_filename)
            with open(label_path, 'r') as f:
                label = int(f.read().strip())
            
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            x = image_tensor.view(1, -1)
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
            data_list.append(data)

    return data_list

def save_new_data(image_tensor, correct_label, save_folder='new_data'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Find the next available index for naming
    index = len([name for name in os.listdir(save_folder) if name.startswith('new_image_')])
    
    # Save the image tensor as a file
    image_path = os.path.join(save_folder, f'new_image_{index}.png')
    transforms.ToPILImage()(image_tensor.squeeze(0)).save(image_path)
    
    # Save the label in a text file
    label_path = os.path.join(save_folder, f'new_image_{index}.txt')
    with open(label_path, 'w') as f:
        f.write(str(correct_label))

# تعیین برچسب از نام فای
def determine_label_from_filename(filename):
    if 'benign' in filename.lower():
        return 0  # Benign class
    elif 'malignant' in filename.lower():
        return 1  # Malignant class
    else:
        return 2  # Normal class

# بارگذاری داده‌ها
def load_data(image_folder, transform):
    data_list = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)  # Convert to tensor
            
            # Convert image to node features
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Simple edge_index for this example
            x = image_tensor.view(1, -1)  # Reshape to graph features (a vector)
            
            label = determine_label_from_filename(filename)
            data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
            data_list.append(data)

    return data_list

def predict(model, image_path, transform):
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    x = image_tensor.view(1, -1)
    
    data = Data(x=x, edge_index=edge_index)
    
    with torch.no_grad():
        output = model(data.to(device))  # Move data to GPU
    
    prediction = output.argmax(dim=1).item()
    return prediction

# Function to process video and make predictions on each frame
def process_video(model, video_path, transform):
    model.eval()
    cap = cv2.VideoCapture(video_path)
    predictions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        x = image_tensor.view(1, -1)
        
        data = Data(x=x, edge_index=edge_index)
        
        with torch.no_grad():
            output = model(data.to(device))
        
        prediction = output.argmax(dim=1).item()
        
        predictions.append(prediction)
        mean_prediction = np.mean(predictions)  # میانگین پیش‌بینی‌ها
        lableres = "result: " + ["Benign", "Malignant", "Normal"][prediction]
        final_prediction = int(np.round(mean_prediction))  # گرد کردن به نزدیک‌ترین عدد صحیح
        finallabel = "final result: " + ["Benign", "Malignant", "Normal"][final_prediction]  #
        # Display the result on the frame
        cv2.putText(frame, lableres, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, finallabel, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    finallabel = ["Benign", "Malignant", "Normal"][final_prediction]
    print("Prediction for the test video: " + finallabel)


def split_dataset(dataset, test_size=0.2):
    # Convert dataset to a list of features and labels
    features = torch.stack([data.x for data in dataset])
    labels = torch.stack([data.y for data in dataset])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    # Create Data objects for train and test sets
    train_data = [Data(x=X_train[i], edge_index=torch.tensor([[0], [0]], dtype=torch.long), y=y_train[i]) for i in range(len(X_train))]
    test_data = [Data(x=X_test[i], edge_index=torch.tensor([[0], [0]], dtype=torch.long), y=y_test[i]) for i in range(len(X_test))]

    return train_data, test_data


def res(status,path):
        
    if status == 1:
        torch.cuda.empty_cache()
        prediction = predict(model,path, transform)
        print(f'Prediction for the test image: {["Benign", "Malignant", "Normal"][prediction]}')
        
        user_feedback = input("Is the prediction correct? (yes/no): ")
        if user_feedback.lower() == 'no':
            correct_label = int(input("Enter the correct label (0: Benign, 1: Malignant, 2: Normal): "))
            image_tensor = transform(Image.open(path).convert('RGB')).unsqueeze(0)
            save_new_data(image_tensor, correct_label)  # Save new data for retraining
            print(f"The prediction was incorrect. Correct label: {['Benign', 'Malignant', 'Normal'][correct_label]}")
            # Load new data for retraining
            new_data = load_new_data()
            if new_data:
                dataset.extend(new_data)
                balanced_dataset = balance_dataset(dataset)
                train_loader = DataLoader(balanced_dataset, batch_size=256, shuffle=True)
                # Lower learning rate for fine-tuning
                optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)  # Reduced learning rate
                criterion = nn.CrossEntropyLoss()
                num_epochs=20
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0

                    for data in train_loader:
                        data = data.to(device)  # Move data to GPU
                        optimizer.zero_grad()  # Zero the gradients

                        output = model(data)  # Forward pass
                        loss = criterion(output, data.y.to(device))  # Compute loss
                        loss.backward()  # Backward pass
                        optimizer.step()  # Update weights

                        running_loss += loss.item()


                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
                torch.save(model.state_dict(), model_path)
                print("Updated model saved.")
        else:
            print("The prediction was correct.")
    
    elif status == 2:
        process_video(model, path, transform)
# اجرای اصلی
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    # Transformations with data augmentation
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ])
    image_folder = 't'  
    dataset = load_data(image_folder, transform)
    balanced_dataset = balance_dataset(dataset)

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = split_dataset(balanced_dataset, test_size=0.2)

    # Create DataLoader for training and testing.
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Define and load model.
    model_path = 'novel_model.pth'
    
    model = NovelConvolutionalNetwork(num_classes=3).to(device)  # Use the new model
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded.")
    except:
        for episode in range(5):
            print("episode: ",episode)
            image_folder = 't'  
            dataset = load_data(image_folder, transform)
            balanced_dataset = balance_dataset(dataset)
            # Create DataLoader for training.
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True) 
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Lower learning rate with regularization

            # Implementing LR scheduler to reduce learning rate on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

            # Training loop
            num_epochs = 15  # Define the number of epochs
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0

                for data in train_loader:
                    data = data.to(device)  # Move data to GPU
                    optimizer.zero_grad()  # Zero the gradients

                    output = model(data)  # Forward pass
                    loss = criterion(output, data.y.to(device))  # Compute loss
                    loss.backward()  # Backward pass
                    optimizer.step()  # Update weights

                    running_loss += loss.item()

                # Step the scheduler
                scheduler.step(running_loss)

                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
            # Save the model after training
        torch.save(model.state_dict(), model_path)
        print("Model saved.")
        with torch.no_grad():
            y_true = []
            y_pred = []
            for data in test_loader:
                data = data.to(device)  # Move data to GPU
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                y_true.extend(data.y.tolist())
                y_pred.extend(predicted.tolist())

            # Print classification report
            print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant', 'Normal']))
        exit()
    # Status input for image or video processing
    while True:
        torch.cuda.empty_cache()
        status = int(input("1 for image prediction, 2 for video processing: "))
        if status in[1,2]:
            path = str(input("please enter yout path: "))
            res(status,path)
        else:
            exit()
import joblib
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import copy
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torchvision import utils
from torchvision.datasets import ImageFolder
from torchsummary import summary
import torch.nn.functional as F
from sklearn.metrics import classification_report
import itertools
from tqdm.notebook import trange
from tqdm.notebook import tqdm
from torch import optim
import warnings
warnings.filterwarnings('ignore')
from io import BytesIO
from flask import Flask, request, jsonify
import requests

from IPython.core.display import display, HTML, Javascript
def image_to_byte_stream(image):
    byte_stream = BytesIO()
    image.save(byte_stream, 'JPEG')  # Save image to the stream in JPEG format
    byte_stream.seek(0)  # Move to the beginning of the stream
    return byte_stream
def image_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')  # or PNG, depending on your needs
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

data_food = pd.read_csv('data_food.csv')

data_food['Protein'] = np.random.uniform(1, 20, size=len(data_food))
data_food['Fat'] = np.random.uniform(1, 15, size=len(data_food))
data_food['Fiber'] = np.random.uniform(1, 10, size=len(data_food))

n_rows = len(data_food)
n_breakfast = n_rows // 3
n_lunch = (n_rows - n_breakfast) // 2
n_dinner = n_rows - n_breakfast - n_lunch

data_food['Meal'] = ['Breakfast'] * n_breakfast + ['Lunch'] * n_lunch + ['Dinner'] * n_dinner

# Add emergency meals
emergency_foods = pd.DataFrame({
    'Food Name': ['Glucose Tablets', 'Orange Juice', 'Honey', 'Candy', 'Sugary Drink'],
    'Carbohydrates': [4, 12, 17, 29, 40],  # Example values
    'Glycemic Index': [100, 50, 61, 65, 68],
    'Meal': ['Emergency'] * 5,
    'Protein': [0, 1, 0, 0, 0],  # Example values
    'Fat': [0, 0, 0, 0, 0],  # Example values
    'Fiber': [0, 0, 0, 0, 0]  # Example values
})

# Combine the relevant features from each dataset
food_data = pd.concat(
    [data_food[['Food Name', 'Carbohydrates', 'Glycemic Index', 'Meal', 'Protein', 'Fat', 'Fiber']], emergency_foods],
    ignore_index=True)

# Group food data by Glycemic Index and calculate the average of new features
food_data_grouped = food_data.groupby('Glycemic Index').agg({
    'Food Name': 'first',
    'Carbohydrates': 'mean',
    'Meal': 'first',
    'Protein': 'mean',
    'Fat': 'mean',
    'Fiber': 'mean'
}).reset_index()

# Load the model from the file
loaded_model = joblib.load('diabetes_meal_plan_model.pkl')
print("Model loaded successfully.")


# Function to generate a meal plan based on glucose level and calculate insulin units
def generate_meal_plan(glucose_level, model, food_data_grouped, insulin_ratio=15, emergency=False):
    """
    Generate a meal plan based on the glucose level and calculate the required insulin units for each meal.
    """
    if emergency:
        # Show emergency foods
        recommended_foods = food_data_grouped[food_data_grouped['Meal'] == 'Emergency']
    else:
        # Prepare the input with the appropriate feature names
        input_data = pd.DataFrame({'Glucose': [glucose_level], 'Protein': [0], 'Fat': [0], 'Fiber': [0]})

        # Predict the required carbohydrates based on the glucose level
        predicted_carbs = model.predict(input_data)[0]

        # Calculate insulin units based on the insulin to carbohydrate ratio
        insulin_units = predicted_carbs / insulin_ratio

        # Filter foods that are close to the predicted carbohydrate value
        recommended_foods = food_data_grouped[np.abs(food_data_grouped['Carbohydrates'] - predicted_carbs) < 5]

    if recommended_foods.empty:
        return {'message': 'No suitable foods found for the given glucose level.'}

    meal_plan = []
    for meal in recommended_foods['Meal'].unique():
        meal_foods = recommended_foods[recommended_foods['Meal'] == meal]
        meal_info = {
            'Meal': meal,
            'Foods': []
        }
        for index, row in meal_foods.iterrows():
            meal_info['Foods'].append({
                'Food Name': row['Food Name'],
                'Carbohydrates': row['Carbohydrates'],
                'Glycemic Index': row['Glycemic Index'],
                'Protein': row['Protein'],
                'Fat': row['Fat'],
                'Fiber': row['Fiber']
            })
        meal_plan.append(meal_info)

    return meal_plan


color_map = ['#FFFFFF','#FF5733']

prompt = color_map[-1]
main_color = color_map[0]
strong_main_color = color_map[1]
custom_colors = [strong_main_color, main_color]

css_file = '''
div #notebook {
background-color: white;
line-height: 20px;
}

#notebook-container {
%s
margin-top: 2em;
padding-top: 2em;
border-top: 4px solid %s;
-webkit-box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5);
    box-shadow: 0px 0px 8px 2px rgba(224, 212, 226, 0.5);
}

div .input {
margin-bottom: 1em;
}

.rendered_html h1, .rendered_html h2, .rendered_html h3, .rendered_html h4, .rendered_html h5, .rendered_html h6 {
color: %s;
font-weight: 600;
}

div.input_area {
border: none;
    background-color: %s;
    border-top: 2px solid %s;
}

div.input_prompt {
color: %s;
}

div.output_prompt {
color: %s; 
}

div.cell.selected:before, div.cell.selected.jupyter-soft-selected:before {
background: %s;
}

div.cell.selected, div.cell.selected.jupyter-soft-selected {
    border-color: %s;
}

.edit_mode div.cell.selected:before {
background: %s;
}

.edit_mode div.cell.selected {
border-color: %s;

}
'''

def to_rgb(h):
    return tuple(int(h[i:i+2], 16) for i in [0, 2, 4])

main_color_rgba = 'rgba(%s, %s, %s, 0.1)' % (to_rgb(main_color[1:]))
open('notebook.css', 'w').write(css_file % ('width: 95%;', main_color, main_color, main_color_rgba,
                                            main_color,  main_color, prompt, main_color, main_color,
                                            main_color, main_color))

def nb():
    return HTML("<style>" + open("notebook.css", "r").read() + "</style>")
nb()



# Define Transformation
transform = transforms.Compose(
    [
        transforms.Resize((255,255)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
   ]
)


# Define object of the Train, Validation, and Test dataset.
train_set = torchvision.datasets.ImageFolder(root='diagonisis of retinopathy data/train', transform=transform)
train_set.transform
val_set = torchvision.datasets.ImageFolder(root='diagonisis of retinopathy data/valid', transform=transform)
val_set.transform
test_set = torchvision.datasets.ImageFolder(root='diagonisis of retinopathy data/test', transform=transform)
test_set.transform


# Import and load Train, Validation and Test set
batch_size = 64

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)



# Print shape of Dataset
for key, value in {'Training data': train_loader, "Validation data": val_loader}.items():
    for X, y in value:
        print(f"{key}:")
        print(f"Shape of X : {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}\n")
        break




    '''This function can be useful in determining the output size of a convolutional layer,
given the input dimensions and the convolutional layer's parameters.'''

def findConv2dOutShape(hin,win,conv,pool=2):
    kernel_size = conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)


# Define Architecture For Retinopathy Model
class CNN_Retino(nn.Module):

    def __init__(self, params):

        super(CNN_Retino, self).__init__()

        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        # CNN Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)

        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self,X):

        X = F.relu(self.conv1(X));
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)



params_model={
        "shape_in": (3,255,255),
        "initial_filters": 8,
        "num_fc1": 100,
        "dropout_rate": 0.15,
        "num_classes": 2}





# Create instantiation of Network class
Retino_model = CNN_Retino(params_model)

# define computation hardware approach (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Retino_model = Retino_model.to(device)


# Model Summary for CNN_Retino
# summary(Retino_model, input_size=(3, 255, 255),device=device.type)

loss_func = nn.NLLLoss(reduction="sum")



opt = optim.Adam(Retino_model.parameters(), lr=1e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)


# Function to get the learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# Function to compute the loss value per batch of data
def loss_batch(loss_func, output, target, opt=None):

    loss = loss_func(output, target) # get loss
    pred = output.argmax(dim=1, keepdim=True) # Get Output Class
    metric_b=pred.eq(target.view_as(pred)).sum().item() # get performance metric

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# Compute the loss value & performance metric for the entire dataset (epoch)
def loss_epoch(model,loss_func,dataset_dl,opt=None):

    run_loss=0.0
    t_metric=0.0
    len_data=len(dataset_dl.dataset)

    # internal loop over dataset
    for xb, yb in dataset_dl:
        # move batch to device
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb) # get model output
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt) # get loss per batch
        run_loss+=loss_b        # update running loss

        if metric_b is not None: # update running metric
            t_metric+=metric_b

    loss=run_loss/float(len_data)  # average loss value
    metric=t_metric/float(len_data) # average metric value

    return loss, metric

params_train={
 "train": train_loader,
 "val": val_loader,
 "epochs": 60,
 "optimiser": optim.Adam(Retino_model.parameters(),lr=1e-4),
 "lr_change": ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1),
 "f_loss": nn.NLLLoss(reduction="sum"),
 "weight_path": "weights.pt"}


epochs=params_train["epochs"]
fig,ax = plt.subplots(1,2,figsize=(12,5))
model = torch.load("Retino_model.pt")

# Move the model to the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# def predict_image(image_path):
#     image = Image.open(image_path)
#     image = transform(image).unsqueeze(0)
#     image = image.to(device)
#     output = model(image)
#     probabilities = torch.softmax(output, dim=1)
#     predicted_class = torch.argmax(probabilities, dim=1)
#     return predicted_class.item()


import torch
import csv
import requests

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'last.pt')

# Load the calorie data
filename = "caloridata - Calorie Dataset.csv"
calorie_data = {}

with open(filename, "r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        calorie_data[row[0]] = float(row[2])


def download_image(url, path):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            return path
        else:
            return None
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        return None


def process_image(image_path):
    results = model(image_path)
    df = results.pandas().xyxy[0]
    search_strings = df["name"].unique().tolist()
    return search_strings


def get_calorie_info(search_strings):
    res = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if any(s in row[0] for s in search_strings):
                result_str = "Calorie of " + row[0] + " in " + row[1] + " grams is " + row[2] + " Calories"
                if result_str not in res:
                    res.append(result_str)
    return res


def calculate_total_calories(foods, quantities):
    total_calories = 0
    for i, food in enumerate(foods):
        quantity = float(quantities[i])
        calories = calorie_data.get(food, 0) * (quantity / 100)
        total_calories += calories
    return total_calories
def predict_image(image_stream):
    try:
        image = Image.open(image_stream)
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class.item()
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def image_to_byte_stream(image):
    byte_stream = BytesIO()
    image.save(byte_stream, 'JPEG')  # Save image to the stream in JPEG format
    byte_stream.seek(0)  # Move to the beginning of the stream
    return byte_stream

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/food', methods=['GET', 'POST'])
def food_model():
    img_url = request.args.get('image_url')
    img_path = download_image(img_url, 'uploaded_image.jpg')

    if not img_path:
        return jsonify({'error': 'Failed to download image'}), 400

    search_strings = process_image(img_path)
    calorie_info = get_calorie_info(search_strings)

    if not search_strings:
        return jsonify({'error': 'Failed to process image'}), 500

    foods = search_strings
    quantities = [100, 150]  # Example quantities; you can modify as needed
    total_calories = calculate_total_calories(foods, quantities)
    insulin_dose = total_calories / 100

    result = {
        'calorie_info': calorie_info,
        'total_calories': total_calories,
        'insulin_dose': insulin_dose
    }
    return jsonify(result)

@app.route('/eye', methods=['GET', 'POST'])
def eye_model():
    try:
        data = request.get_json()
        app.logger.info(f"Received data: {data}")
        if not data or 'image' not in data:
            app.logger.warning("No image provided in the request")
            return jsonify({'error': 'No image provided'}), 400

        image_data = data['image']
        app.logger.info("Decoding base64 image data")
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        app.logger.info("Image decoded successfully")

        # Convert image to a byte stream before processing if needed
        image_stream = image_to_byte_stream(image)
        app.logger.info("Image converted to byte stream")

        predicted_image = predict_image(image_stream)  # Make sure this function is correctly using a byte stream
        app.logger.info(f"Prediction result: {predicted_image}")

        # Example response based on prediction
        if predicted_image == 0:
            prediction_result = 'DR'
        else:
            prediction_result = 'No_DR'

        return jsonify({'prediction': prediction_result})
    except Exception as e:
        app.logger.error(f"Error processing image: {e}", exc_info=True)
        return jsonify({'error': f'Failed to process images: {str(e)}'}), 500


@app.route('/sugar', methods=['GET', 'POST'])
def sugar_model():
    glucose_level = request.args.get('glucose_level')

    if not glucose_level:
        return jsonify({'error': 'Missing glucose level'}), 400

    try:
        glucose_level = float(glucose_level)
    except ValueError:
        return jsonify({'error': 'Invalid glucose level'}), 400

    result = generate_meal_plan(glucose_level, loaded_model, food_data_grouped)

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

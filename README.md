# AIT-GPT demo

## Overview

   Welcome to the AIT-GPT Demo App! This web-based application demonstrates the basic functionality of langchain model, allowing users to achieve the comparable information extraction performance but with an input of student handbook of AIT from [here](https://drive.google.com/file/d/1sKaWzNwMK1_rPUIRGWl9kuNzK4qAjXei/view).

## Pretrained model and Dataset

   The AIT-GPT is Retrieval-Augmented Generation (RAG) which is a LLM that will answer the question based on the input data (in this case, [student handbook of AIT](https://drive.google.com/file/d/1sKaWzNwMK1_rPUIRGWl9kuNzK4qAjXei/view)) using [Fastchat Model](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) as the RAG model. 

## Features

   - **Input Query:** User can enter a query to ask about AIT.
   - **Submit Button:** User clicks *submit* after typing the query. The app will search through the internal document about the question and its proper answer.

## Application

### Prerequisites

- Make sure you installed git and have GPU available. A Cuda is needed for this application.

### Set-up wsl with cuda

1. Install wsl on your Windows. If you use Linux, you can skip to step 3.:

   ```powershell
   wsl --install
   ```

2. Access the ubuntu terminal

    ```powershell
   wsl
   ```
 
3. Check if you have GPU available and run the following command line to install python, cuda, and torch

    ```terminal
   sudo apt update && sudo apt upgrade
   sudo apt install python3 python3-pip
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
### Installation

1. Clone the repository:

   ```terminal
   cd ~
   git clone https://github.com/thassung/AIT-GPT_demo.git
   ```

2. Install the required Python dependencies:

   ```bash
   cd AIT-GPT_demo
   pip3 install -r requirements.txt
   cd model
   git clone https://huggingface.co/lmsys/fastchat-t5-3b-v1.0
   ```

3. Navigate to the app directoty and run the app:
   ```bash
   cd ../app
   ```

4. Start the flask application:
   ```bash
   python3 main.py
   ```

   You can access the application via http://localhost:8080

   Below is how the app should look like.


## About the Model Performance



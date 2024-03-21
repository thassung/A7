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

1. Install wsl on your Windows via powershell. If you use Linux, you can skip to step 3.:

   ```powershell
   wsl --install
   ```

2. Access the ubuntu terminal:

    ```powershell
   wsl
   ```
 
3. Check if you have GPU available and run the following command line to install python, cuda, and torch:

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

   ![image](https://github.com/thassung/AIT-GPT_demo/assets/105700459/205b6f20-2185-4d34-8190-8f5ac45fc493)


## Comment about the Model Performance

So far, with sample question in this ipynb, the model can answer the question correctly though it is might not be the best detailed answer. For example, when asked "where is AIT?", it answers with "P.O.  Box  4  Klong  Luang  Pathumthani  12120,  Thailand" which is correct but not the best, accurated answer you can get from the document. The model does not exhibit an AI hallucination. Or when it is asked how many students and faculties are in AIT, it answers "1200+ from 40+ countries" which is the number of students and not accounted for faculties. It is obvious that an information is missing (see above figure). When asked who the AIT president is which is an information not mentioned in the document, the model answers with "the  text  does  not  provide  any  information  about  AIT's  president.  The  President  is  the  head  of  the  AIT  Student  Union,  which  is  responsible  for  implementing  policies  and  ensuring  the  well-being  of  the  students.  The  Vice  President  becomes  President  of  the  SU  in  the  succeeding  semester." Although there is no hallucination in the last sentence, most of the answer is not directly related and irrelevant to the question. 



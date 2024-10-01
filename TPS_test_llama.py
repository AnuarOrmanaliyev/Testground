import transformers
import torch
import time
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# For multi-GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize Accelerate
from accelerate import Accelerator
accelerator = Accelerator()

import wandb

#wandb.init(project = 'TPS Tests', entity = 'anuar-ormanaliyev-q19')

token = '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'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(24)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token = token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.float16,token = token)

model = accelerator.prepare(model)
#device_map = infer_auto_device_map(model)

#model = load_checkpoint_and_dispatch(model, device_map=device_map)


model.to(device)

model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]


# Define and log hyperparameters
config = {
    "model_name": "llama-3.1-8b-instruct",
    "max_new_tokens": 100,
    'number fo prompts': 5,
    'warmup': 'no',
    'input_tokens_counted': 'yes',
    'number of gpus' : '1',
    'top_p' : 0.1,
    'temperature' : 0.1,
    'torch_dtype': 'float16'
}

#wandb.config.update(config)

# Define a sample input prompt for testing
results = dict()

custom_instruction3 = """
You are a model that, given a text request from a real user, generates an API request of the following form:
{
  "action": "some_action"
}

Action field can only be one of following type:
["psc_appointment", "request_recipient", "request_region", "select_region","get_license"];

Your role is to decide the appropriate action based on the user's input. Only include the "action" field unless the example shows additional fields.

Do not add fields or modify the structure unless instructed. If the selected action has additional fields (e.g., "region", "phone") then add them into "params" as shown int he examples below.
Refer to the examples below to understand how to structure the response.
Return all answers in latin alphabet. If the input is cyrillic still return latin.
Look cyrillic to latin transformation in the examples below

Examples:

Request: Бронирование очереди в ЦОН
Response:
{
  "action": "psc_appointment"
}

Request: На кого хотите забронировать очередь в отдел ЦОН?
Response:
{
  "action": "request_recipient"
}

Request: Пожалуйста, подскажите область
Response:
{
  "action": "request_region"
}

Request: город астана
Response:
{
  "action": "select_region",
  "params": {
    "region": "astana"
  }
}

Request: Получить права
Response:
{
  "action": "get_license"
    
}

Request: Пожалуйста, укажите ваш идентификатор пользователя
Response:
{
"action": "request_user_id"
}

Request: Мой идентификатор пользователя — 12345
Response:
{
  "action": "provide_user_id",
  "params": {
    "user_id": "12345"
  }
}

Request: Меня зовут Джон Доу
Response:
{
  "action": "provide_name",
  "params": {
    "name": "John Doe"
  }
}

Request: Пожалуйста, укажите ваши контактные данные
{
  "action": "request_contact_info"
}

Request: Мой email john.doe@example.com и телефон 123-456-7890
Response:
{
  "action": "provide_contact_info",
  "params": {
    "contact_info": {
      "email": "john.doe@example.com",
      "phone": "123-456-7890"
    }
  }
}



Request: "цонға жазылу керек еді удостоверение жасауға ертеңге ақтоғайда ербол атым"
Response:
{
 "action": "psc_appoinment",
 "params":{
 "contact_info": {  
   "name": "Yerbol",
   "region" "Aqtogay",       
  }
 }
}
}

Request: 'ануар лесов хочу удостак сделать запишите на 29ое сентября в алмате'
Response:

Response:
{
 "action": "psc_appoinment",
 "params":{
 "contact_info": {  
   "name": "Anuar Lesov",
   "region": "Almaty",  
   "date" : "29 september"     
  }
 }
}
}
Request: "" удостоверение жоғалтып едім жаңасын жасату керек ескі удостоверение номер 111223 телефоным 777888999б жаңатаста тұрам Жахаңгір атым, цонға бейсенбіге сағат үшке жазшы""

Response:
{
 "action": "psc_appoinment",
 "params":{
 "contact_info": {  
   "name": "Zhakhanggir",
   "region": "Zhanatas",
   "phone": "777888999", 
  }
 }
}
}

Request: "атырау қаласында 30 август күні права жасату керек жалбыз женешев"

Response:
{
  "action": "get_license",
  "params": {
    "contact_info": {
      "name": "Zhalbyz Zheneshev",
      "region": "Atyrau",
      "date": "30 August",
    }
  }
}

Request: "15 мартқа Құлсары ауылында праваға жащищ. Ерік Ерімбетов удостак 7845161152 сотка 84123165"

Response:
{
  "action": "get_license",
  "params": {
    "contact_info": {
      "name": "Yerik Yerimbetov",
      "region": "Qulsary",
      "user_id": "7845161152",
      "date": "15 March",
      "phone": "84123165",
    }
  }
}

Request: "мен алматы қакаласында тұрам бірақ тараз қаласынанда цонға жазылу керек боп тұр удостак 77777 телефон 555555 менің атым Мөдліп Маентова"

Response:
{
  "action": "psc_appointment",
  "params": {
    "contact_info": {
      "name": "Mödlip Mayentova",
      "region": "Taraz",
      "user_id": "77777",
      "phone": "555555",
    }
  }
}

Request: "права алуға 20 қантарда маметов кошесинде керек боп тур. номерім 5478 удостак номер 98745 жанабала есжанов ақбұдак қаласы"
Response: 
{
  "action": "get_license",
  "params": {
    "contact_info": {
      "name": "Yeszhanov Zhanabala",
      "region": "Aqbudak",
      "user_id": "98745",
      "date": "20 January",
      "phone": "5478",
    }
  }
}

Request: " минде бір сурақ бар ет көкменир деген жерде права алсам тек телефон нөмир пойдет па номер 75412 удошка 1236454па папам сарува лмаива деген ымя паставил"
Response:
{
  "action": "get_license",
  "params": {
    "contact_info": {
      "name": "Saruva Lmaiva",
      "region": "Kökmenir",
      "user_id": "1236454",
      "phone": "75412",
    }
  }
}

Request:
"есімім Прат78 удо 14587цу телефон вө-87124ю цонға жаха салщы ьисинш априль куни"
Response:
{
  "action": "psc_appointment",
  "params": {
    "contact_info": {
      "name": "Prat",
      "user_id": "14587",
      "phone": "87124",
      "date": "5 April"
    }
  }
}

Request: "сағат 9да июнпын 20чы күні црнға удостак алу үшн жазылу керек еск удостак номер 45871 телефон 78452 магзум тастеректегы цон"


Response:
{
  "action": "psc_appointment",
  "params": {
    "contact_info": {
      "name": "Magzum",
      "region": "Tasterek",
      "phone": "78452",
      "date": "20 June 9:00"
    }
  }
}

Request:
"правани бесауыл жерине алыр келд мен өзм алатаякта турам сол бесауылға жаза салщ асанали имукатов удо номер 147258 телефоныа 963852 15щы декабоь"

Response:
{
  "action": "get_license",
  "params": {
    "contact_info": {
      "name": "Asanali Imukatov",
      "region": "Besauyl",
      "user_id": "147258",
      "phone": "963852",
      "date": "15 December"
    }
  }
}
"""



i =0

prompts = [
    "нужно записаться в цон на 6ое ноября в город аксу мое имя мауленов магид мой номер телефона 777888999?",
    "хочу получить права в городе алмаьы 19го апреля номер удо 666555444?",
    'справка керег ед цоннан алмас щакуновка щучинскада ьурам мен 1 янвалда',
    'горож караганда номер удостоверения личности 111222 мобидбный 444555 имя рлжас ескеибетов нужна зпписаь в цон на 8ое июля',
    'корғалжында абай ауданында мират атым права алу керек төртінші ой бксінші майда'
]

for prompt in prompts:    
    # Tokenize the input prompt
    #batch_size = 64
  
    input_ids = tokenizer(prompt, return_tensors="pt")
    
    input_ids = {k: v.to(accelerator.device) for k, v in input_ids.items()}

    # Set the number of new tokens to generate (modify as needed for testing)
    max_new_tokens = 100

    #WarmUp 
    model.generate(**input_ids, max_new_tokens=max_new_tokens, temperature = 0.1, top_p = 0.1)
    
    #Start the timer
    start_time = time.time()

    with accelerator.autocast():
        output = model.generate(**input_ids, max_new_tokens=max_new_tokens, temperature = 0.1, top_p = 0.1)
  
    # Stop the timer
    end_time = time.time()

    # Calculate the time taken in seconds
    time_taken = end_time - start_time

    # Calculate the total number of tokens generated
    total_tokens =  output.shape[-1]# Includes both input and output tokens

    # Calculate Tokens per Second (TPS)
    tps = total_tokens / time_taken
    
    #Decoded text
    decoded_text = tokenizer.decode(output.tolist()[0], skip_special_tokens=True)

    with open("input_data"+str(i)+".txt", "w") as f:
        f.write(decoded_text)

    artifact = wandb.Artifact("output"+str(i)+".txt", type = "result")
    artifact.add_file("input_data"+str(i)+".txt")
    
    #wandb.log_artifact(artifact)


    result = {
        "Tokens per Second": tps,
            'Input':prompt,
    }
    results['dict_'+ str(i)] = result

    i+=1

    print(tps)


#Log the TPS to WandB
#wandb.log(results)

#wandb.finish()
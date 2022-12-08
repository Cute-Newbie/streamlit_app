# 프롬프트 창에 
# streamlit run app.py 입력 
# 저장해 가면서 앱 만들기  

import streamlit as st 
from streamlit_chat import message

#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwiase import cosime_similarity
import json
import numpy as np
import pandas as pd
import torch
#from pytorch_lightning import Trainer
#from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.core.lightning import LightningModule
#from torch.utils.data import DataLoader, Dataset
#from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast #GPT2LMHeadModel
import re
import joblib



#########################################################
# for KORGPT-2
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

########################################
#For KorBert


#####################################################################


# 필요한 함수 정의 


koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 




model = joblib.load('model_2.pkl')
# emotion_model = joblib.load('emotion_model.pkl')

st.header('심리상담 챗봇')
st.markdown("챗봇")


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []




with st.form('form',clear_on_submit = True):
    user_input = st.text_input('당신: ','')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    a=""
    while 1:
        input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + user_input + SENT + A_TKN + a)).unsqueeze(dim=0)
        pred = model(input_ids)
        pred = pred.logits
        
        gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
        if gen == EOS:
            break
        a += gen.replace("▁", " ")

    st.session_state.past.append(user_input)
    st.session_state.generated.append(a)

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i],
    is_user = True,
    key = str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i],key = str(i) + '_bot')
   








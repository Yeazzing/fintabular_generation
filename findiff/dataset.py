import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from torch.utils.data import TensorDataset, random_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class FinancialDataset:
    def __init__(self, path, cat_emb_dim=2, label_cols=['LifeStage'], val_ratio=0.2, random_state=1234):
        self.path = path
        self.cat_emb_dim = cat_emb_dim
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.label_cols = label_cols
        
        self.train_df = None
        self.val_df = None
        
        self.cat_attrs = ['회원여부이용가능', '회원여부이용가능CA', '회원여부이용가능카드론','동의여부한도증액안내', '수신거부여부TM', '수신거부여부DM',
       '수신거부여부SMS','마케팅동의여부', '1순위신용체크구분', '보유여부해외겸용본인', '이용가능여부해외겸용본인', '이용여부3M해외겸용본인', '보유여부해외겸용신용본인',
       '이용가능여부해외겸용신용본인', '이용여부3M해외겸용신용본인', '연령']
        
        self.num_attrs = [
            '소지카드수유효신용', '소지카드수이용가능신용', '입회일자신용',
            '탈회횟수누적', '탈회횟수발급1년이내', '유효카드수신용체크', '유효카드수신용',
            '유효카드수신용가족', '유효카드수체크', '이용가능카드수신용체크', '이용가능카드수신용',
            '이용카드수신용체크', '이용카드수신용', '이용카드수체크', '이용금액R3M신용체크',
            '이용금액R3M신용', '이용금액R3M체크', '1순위카드이용금액', '1순위카드이용건수',
            '2순위카드이용금액', '카드신청건수'
        ]
        
        self.num_scalers = {}
        self.label_encoder = None
        self.condition_encoders = {}
        self.vocab_per_attr = {}
        self.vocab_per_condition = {}
        self.n_cat_tokens = None
        self.encoded_dim = None
        self.n_classes = None
        self.cat_dim = None


    def prepare_data(self):
        df = pd.read_csv(self.path)
        df.columns = [col.replace('_', '') for col in df.columns]
        
        #결측치 처리
        df = df.replace('_', np.nan)
        df['1순위신용체크구분'] = df['1순위신용체크구분'].fillna('누락') 
        df = df.dropna(axis=1)

        for cat_attr in self.cat_attrs:
            df[cat_attr] = cat_attr + '_' + df[cat_attr].astype(str)
            
        train = df[[*self.cat_attrs, *self.num_attrs]]
        
        # 수치형 변환
        train_num = train[self.num_attrs].copy()
        for num_attr in self.num_attrs:
            QT = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
            train_num[num_attr] = QT.fit_transform(train[[num_attr]])
            self.num_scalers[num_attr] = QT
        
        #범주형 변환
        train_cat = train[self.cat_attrs].copy() #.drop(columns=self.label_cols, axis=1)
        #self.cat_attrs = [c for c in self.cat_attrs if c not in self.label_cols]
        
        vocabulary_classes = np.unique(train_cat)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(vocabulary_classes)
        train_cat = train_cat.apply(self.label_encoder.transform)
        self.vocab_per_attr = {cat_attr: set(train_cat[cat_attr]) for cat_attr in self.cat_attrs}
        
        #condition 컬럼 변환          
        label = df[self.label_cols].copy()  
        if len(self.label_cols) > 1:
            for label_col in self.label_cols:
                encoder = LabelEncoder()
                label[label_col] = encoder.fit_transform(label[label_col])
                self.condition_encoders[label_col] = encoder
                self.vocab_per_condition[label_col] =encoder.classes_
        else:
            # 단일 컬럼일 경우
            label_col = self.label_cols[0]
            encoder = LabelEncoder()
            label[label_col] = encoder.fit_transform(label[label_col])
            label = label.squeeze()
            self.condition_encoders[label_col] = encoder
        
        # 텐서 변환
        train_num_torch = torch.tensor(train_num.values, dtype=torch.float32)
        train_cat_torch = torch.tensor(train_cat.values, dtype=torch.long) #.contiguous()
        label_torch = torch.tensor(label.values, dtype=torch.long)  #.contiguous()
        
        full_dataset = TensorDataset(train_cat_torch, train_num_torch, label_torch)
        
        # train / val split
        generator = torch.Generator().manual_seed(self.random_state)  
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * self.val_ratio)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        # eval용 데이터프레임 저장 (전처리 되기 전 데이터 저장)
        train_indices = train_dataset.indices
        train_df = train.iloc[train_indices].reset_index(drop=True)
        self.train_df = train_df
        train_df.to_csv(os.path.join(BASE_DIR, "train.csv"), index=False)
        
        val_indices = val_dataset.indices
        val_df = train.iloc[val_indices].reset_index(drop=True)
        self.val_df = val_df
        val_df.to_csv(os.path.join(BASE_DIR, "eval.csv"), index=False)
        
        #dim set
        self.n_cat_tokens = sum(len(v) for v in self.vocab_per_attr.values())
        self.cat_dim = self.cat_emb_dim * len(self.cat_attrs)
        num_dim = len(self.num_attrs)
        self.encoded_dim = self.cat_dim + num_dim
        self.n_classes = len(encoder.classes_) if len(self.label_cols) == 1 else [len(self.vocab_per_condition[col]) for col in self.label_cols]  #1개일때-정수, 다중일때-리스트
        
        return train_dataset, val_dataset


if __name__ == "__main__":
    datapreparing = FinancialDataset(path='/workspace/data/201807_회원정보.csv') 
    traindata, valdata = datapreparing.prepare_data()
    len(traindata)
    len(valdata)
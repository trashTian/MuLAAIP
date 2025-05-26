"""
PLM: Esm2_150M, ProtBert
ALM: Ablang, AntiBERTy, BERT2DAb
"""
import multiprocessing

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import esm
import os


def extract_from_ProtTrans():
    df = pd.read_csv('7_26_mutate_resample.csv')

    affinity = df['delta_g'].values
    # df['sequence1'] = df['Sequence']
    # df['sequence2'] = df['Target']
    # df['sequence1'] = df['seq_ab']
    # df['sequence2'] = df['seq_ag']
    # affinity = df['delta_g'].values
    df['sequence1'] = df['antibody_Hchain_sequence'].fillna('') + df['antibody_Lchain_sequence'].fillna('')
    df['sequence2'] = df['antigen_sequence']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
    model.full() if device == 'cpu' else model.half()

    features = []

    for i in range(0, df.shape[0]):
        print(i)
        # prepare your protein sequences as a list
        sequence_examples = [df['sequence1'][i], df['sequence2'][i]]

        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

        ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        emb_0 = embedding_repr.last_hidden_state[0, :len(df['sequence1'][i])]  # shape (sequence_length x 1024)
        # embedding_path = r"E:\ProgrammingSpace\Gitee\data\DeepAntibody\ProtTrans_embeddings\{}.npy".format(str(df['Index'][i]))
        # np.save(embedding_path, emb_0.cpu().numpy())
        emb_1 = embedding_repr.last_hidden_state[1, :len(df['sequence2'][i])]

        emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)
        emb_1_per_protein = emb_1.mean(dim=0)

        feature1 = emb_0_per_protein.cpu().numpy()
        feature2 = emb_1_per_protein.cpu().numpy()

        feature = np.concatenate((feature1, feature2))
        features.append(feature)

    current_path = Path.cwd()
    embedding_path = current_path.joinpath('mutate_embeddings')
    embedding_file = embedding_path / 'ProtTrans_mutate_726.npy'
    affinity_file = embedding_path / 'affinity_726.npy'
    np.save(str(embedding_file), np.array(features))
    np.save(str(affinity_file), affinity)


def extract_from_ESM2_650M():
    """
    ESM-2能处理的最大蛋白质序列长度为1024，超过1024的序列需要切片处理
    embedding为1280 DIM
    :return:
    """
    df = pd.read_csv('7_26_mutate_resample.csv')
    # df['sequence1'] = df['Sequence']
    # df['sequence2'] = df['Target']
    # df['sequence1'] = df['seq_ab']
    # df['sequence2'] = df['seq_ag']
    df['sequence1'] = df['antibody_Hchain_sequence'].fillna('') + df['antibody_Lchain_sequence'].fillna('')
    df['sequence2'] = df['antigen_sequence']

    def split_sequences(s, max_length=1024):
        if len(s) <= max_length:
            return [('s1', s)]
        else:
            num_splits = len(s) // max_length
            splits = []
            for i in range(num_splits):
                start = i * max_length
                end = (i + 1) * max_length
                splits.append(('s{}'.format(i + 1), s[start:end]))

            remaining = len(s) % max_length
            if remaining > 0:
                splits.append(('s{}'.format(num_splits + 1), s[-remaining:]))

            return splits

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    antigen_features = []
    antibody_features = []

    for i in range(0, df.shape[0]):
        print(i)

        # =================================================================================================
        # 提取抗原特征
        antigens = split_sequences(df['sequence2'][i])
        # 判断切片长度
        if len(antigens) < 2:
            batch_labels, batch_strs, batch_tokens = batch_converter(antigens)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                batch_tokens = batch_tokens.to(device)
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

            antigen_features.append(sequence_representations[0].cpu().numpy())
        # 遍历切片list
        else:
            representations = []

            for antigen in antigens:
                # 装箱
                _, _, batch_tokens = batch_converter([antigen])
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    batch_tokens = batch_tokens.to(device)
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]

                sequence_representations = []
                for i, tokens_len in enumerate(batch_lens):
                    sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

                representations.append(sequence_representations[0])

            # 初始化总和矩阵
            sum = torch.zeros_like(representations[0])
            for t in representations:
                sum += t

            average_representation = sum / len(representations)
            print(average_representation.shape)
            antigen_features.append(average_representation.cpu().numpy())

        # ===============================================================================================
        # 提取抗体特征
        antibodies = split_sequences(df['sequence1'][i])
        # 判断切片长度
        if len(antibodies) < 2:
            batch_labels, batch_strs, batch_tokens = batch_converter(antibodies)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                batch_tokens = batch_tokens.to(device)
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
            antibody_features.append(sequence_representations[0].cpu().numpy())
        # 遍历切片list
        else:
            representations = []

            for antibody in antibodies:
                # 装箱
                _, _, batch_tokens = batch_converter([antibody])
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    batch_tokens = batch_tokens.to(device)
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33]

                sequence_representations = []
                for i, tokens_len in enumerate(batch_lens):
                    sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

                representations.append(sequence_representations[0])

            # 初始化总和矩阵
            sum = torch.zeros_like(representations[0])
            for t in representations:
                sum += t

            average_representation = sum / len(representations)
            antibody_features.append(average_representation.cpu().numpy())

    # ===============================================================================================
    # 拼接抗原和抗体的特征
    # print(antigen_features)
    # print(len(antigen_features))
    # print(len(antigen_features[0]))
    #
    # print('=====')
    # print(len(antibody_features))
    # print(antibody_features)
    # print(len(antibody_features[0]))

    features = np.concatenate((antibody_features, antigen_features), axis=1)
    print(features.shape)
    current_path = Path.cwd()
    embedding_path = current_path.joinpath('mutate_embeddings')
    embedding_file = embedding_path / 'ESM2_mutate_726.npy'
    print(embedding_file)
    np.save(str(embedding_file), np.array(features))


def extract_from_AbLang():
    """
    length_a_sequence * 480
    :return:
    """
    import ablang2
    ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, device='cuda')
    df = pd.read_csv('7_26_mutate_resample.csv')
    # df['sequence1'] = df['Sequence']
    # df['sequence2'] = df['Target']
    # df['sequence1'] = df['seq_ab']
    # df['sequence2'] = df['seq_ag']
    # affinity = df['delta_g'].values
    df['sequence1'] = df['antibody_Hchain_sequence'].fillna('') + df['antibody_Lchain_sequence'].fillna('')
    df['sequence2'] = df['antigen_sequence']
    # Download and initialise the model
    antibody_feature = []
    antigen_feature = []
    for i in range(0, df.shape[0]):
        seq = [df['sequence1'][i]]

        # Tokenize input sequences
        tokenized_seq = ablang.tokenizer(seq, pad=True, w_extra_tkns=False, device="cuda")

        # Generate rescodings
        with torch.no_grad():
            rescoding = ablang.AbRep(tokenized_seq).last_hidden_states

        rep = torch.mean(rescoding[0], dim=0, keepdim=True)
        antibody_feature.append(rep.cpu().numpy()[0])

        # print(rescoding)
        # print(rescoding.shape)  #
        # print(rescoding[0].shape)

    for i in range(0, df.shape[0]):
        print(i)
        seq = [df['sequence2'][i]]

        # Tokenize input sequences
        tokenized_seq = ablang.tokenizer(seq, pad=True, w_extra_tkns=False, device="cuda")

        # Generate rescodings
        with torch.no_grad():
            rescoding = ablang.AbRep(tokenized_seq).last_hidden_states

        rep = torch.mean(rescoding[0], dim=0, keepdim=True)
        antigen_feature.append(rep.cpu().numpy()[0])

    features = np.concatenate((antibody_feature, antigen_feature), axis=1)
    print(features.shape)

    current_path = Path.cwd()
    embedding_path = current_path.joinpath('mutate_embeddings')
    embedding_file = embedding_path / 'AbLang_mutate_726.npy'

    np.save(str(embedding_file), np.array(features))


def extract_from_AntiBERTy():
    """
        (length_a_sequence + 2) * 512
        :return:
    """
    from antiberty import AntiBERTyRunner

    df = pd.read_csv('7_26_mutate_resample.csv')
    # df['sequence1'] = df['Sequence']
    # df['sequence2'] = df['Target']
    # df['sequence1'] = df['seq_ab']
    # df['sequence2'] = df['seq_ag']
    # affinity = df['delta_g'].values
    df['sequence1'] = df['antibody_Hchain_sequence'].fillna('') + df['antibody_Lchain_sequence'].fillna('')
    df['sequence2'] = df['antigen_sequence']
    antiberty = AntiBERTyRunner()

    antibody_feature = []
    antigen_feature = []

    for i in range(0, df.shape[0]):
        print(i)
        if len(df['sequence1'][i]) > 510:
            sequences = [df['sequence1'][i][:510]]
        else:
            sequences = [df['sequence1'][i]]
        embeddings = antiberty.embed(sequences)

        # print(embeddings)
        # print(len(embeddings)) # len of sequences
        # print(embeddings[0].shape)
        # print(embeddings[0])

        average_embeddings = torch.mean(embeddings[0], dim=0)
        # print(average_embeddings.shape)
        antibody_feature.append(average_embeddings.cpu().numpy())

    for i in range(0, df.shape[0]):
        print(i)

        if len(df['sequence2'][i]) <= 512:
            sequences = [df['sequence2'][i]]
            embeddings = antiberty.embed(sequences)
            average_embeddings = torch.mean(embeddings[0], dim=0)
            antigen_feature.append(average_embeddings.cpu().numpy())

        else:
            num_segments = len(df['sequence2'][i]) // 512
            remainder = len(df['sequence2'][i]) % 512
            sequences = []
            for i in range(num_segments):
                sequences.append(df['sequence2'][i][i * 512: (i + 1) * 512])
            if remainder > 0:
                sequences.append(df['sequence2'][i][num_segments * 512:])
            temp = []
            for sequence in sequences:
                embeddings = antiberty.embed([sequence])
                average_embeddings = torch.mean(embeddings[0], dim=0)
                temp.append(average_embeddings.cpu().numpy())
            antigen_feature.append(np.mean(np.array(temp), axis=0))

    features = np.concatenate((antibody_feature, antigen_feature), axis=1)
    print(features.shape)

    current_path = Path.cwd()
    embedding_path = current_path.joinpath('mutate_embeddings')
    embedding_file = embedding_path / 'AntiBERTy_mutate_726.npy'

    np.save(str(embedding_file), np.array(features))


def extract_from_BERT2DAb():
    from transformers import BertTokenizer, BertModel
    import ast
    df = pd.read_csv(r'D:\wild_89.csv')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # =============================================================================================
    # 提取重链特征
    tokenizer_H = BertTokenizer.from_pretrained("w139700701/BERT2DAb_H")
    model_H = BertModel.from_pretrained("w139700701/BERT2DAb_H")
    model_H.to(device)

    hchains = []
    for i in range(0, df.shape[0]):
        print(i)
        if pd.isnull(df['a'][i]):
            # 没有轻链则填充空
            hchains.append(np.zeros(768))
        else:

            H_chain = ast.literal_eval(df['a'][i])
            if len(H_chain) <= 126:
                encoded_input = tokenizer_H.encode_plus(
                    H_chain,
                    padding=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                # 将编码后的文本数据转换为张量并移动到设备上
                input_ids = encoded_input["input_ids"].to(device)
                # print(input_ids)
                attention_mask = encoded_input["attention_mask"].to(device)
                # 获取模型的输出（嵌入向量）
                with torch.no_grad():
                    outputs = model_H(input_ids, attention_mask=attention_mask)

                # 获取嵌入向量
                embeddings = outputs.last_hidden_state
                # print('================={}============='.format(embeddings.shape))
                # print(embeddings)
                # print(embeddings.shape)
                # 嵌入的第一行与最后一行是起始子和终止子
                average_embeddings = torch.mean(embeddings[0], dim=0, keepdim=True)
                # print(average_embeddings)
                # print(average_embeddings.shape)

            else:
                A_chains = [H_chain[i:i + 126] for i in range(0, len(H_chain), 126)]
                embeds = []
                for chain in A_chains:
                    encoded_input = tokenizer_H.encode_plus(
                        chain,
                        padding=True,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
                    input_ids = encoded_input["input_ids"].to(device)
                    attention_mask = encoded_input["attention_mask"].to(device)
                    with torch.no_grad():
                        outputs = model_H(input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state
                    average = torch.mean(embeddings[0], dim=0, keepdim=True)
                    embeds.append(average)
                average_embeddings = torch.mean(torch.stack(embeds, dim=0), dim=0)
            hchains.append(average_embeddings[0].detach().cpu().numpy())
    #
    # # =============================================================================================
    # 提取轻链特征
    lchains = []
    tokenizer_L = BertTokenizer.from_pretrained("w139700701/BERT2DAb_L")
    model_L = BertModel.from_pretrained("w139700701/BERT2DAb_L")
    model_L.to(device)
    for i in range(0, df.shape[0]):
        print(i)
        if pd.isnull(df['b'][i]):
            print('=============================================')
            # 没有轻链则填充空
            lchains.append(np.zeros(768))
        else:
            L_chian = ast.literal_eval(df['b'][i])
            if len(L_chian) <= 126:
                encoded_input = tokenizer_L.encode_plus(
                    L_chian,
                    padding=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                input_ids = encoded_input["input_ids"].to(device)
                attention_mask = encoded_input["attention_mask"].to(device)
                with torch.no_grad():
                    outputs = model_L(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state
                average_embeddings = torch.mean(embeddings[0], dim=0, keepdim=True)

            else:
                A_chains = [L_chian[i:i + 126] for i in range(0, len(L_chian), 126)]
                embeds = []
                for chain in A_chains:
                    encoded_input = tokenizer_L.encode_plus(
                        chain,
                        padding=True,
                        add_special_tokens=True,
                        return_tensors="pt"
                    )
                    input_ids = encoded_input["input_ids"].to(device)
                    attention_mask = encoded_input["attention_mask"].to(device)
                    with torch.no_grad():
                        outputs = model_L(input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state
                    average = torch.mean(embeddings[0], dim=0, keepdim=True)
                    embeds.append(average)
                average_embeddings = torch.mean(torch.stack(embeds, dim=0), dim=0)
            lchains.append(average_embeddings[0].detach().cpu().numpy())

    # =============================================================================================
    # 提取抗原特征
    tokenizer_A = BertTokenizer.from_pretrained("w139700701/BERT2DAb_H")
    model_A = BertModel.from_pretrained("w139700701/BERT2DAb_H")
    model_A.to(device)

    achains = []
    for i in range(0, df.shape[0]):
        print(i)  # 80
        A_chain = ast.literal_eval(df['c'][i])

        if len(A_chain) <= 126:
            # print(len(A_chain))

            encoded_input = tokenizer_A.encode_plus(
                A_chain,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

            input_ids = encoded_input["input_ids"].to(device)
            # print(input_ids)
            # print(len(input_ids))
            attention_mask = encoded_input["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model_A(input_ids, attention_mask=attention_mask)

            embeddings = outputs.last_hidden_state

            average_embeddings = torch.mean(embeddings[0], dim=0, keepdim=True)


        else:
            A_chains = [A_chain[i:i + 126] for i in range(0, len(A_chain), 126)]
            embeds = []
            for chain in A_chains:
                encoded_input = tokenizer_A.encode_plus(
                    chain,
                    padding=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )

                input_ids = encoded_input["input_ids"].to(device)

                attention_mask = encoded_input["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = model_A(input_ids, attention_mask=attention_mask)

                embeddings = outputs.last_hidden_state

                average = torch.mean(embeddings[0], dim=0, keepdim=True)
                embeds.append(average)
            average_embeddings = torch.mean(torch.stack(embeds, dim=0), dim=0)

        # print(average_embeddings)
        achains.append(average_embeddings[0].detach().cpu().numpy())
    # =============================================================================================

    features = np.concatenate((hchains, lchains, achains), axis=1)
    print(features.shape)

    current_path = Path.cwd()
    embedding_path = current_path.joinpath('wild_embeddings')
    embedding_file = embedding_path / 'BERT2DAb_features810.npy'
    affinity_file = embedding_path / 'affinity_89.npy'
    affinity = df['delta_g'].values
    np.save(str(affinity_file), affinity)
    np.save(str(embedding_file), np.array(features))


def compare_different_embeddings():
    current_path = Path.cwd()
    embedding_path = current_path.joinpath('alphaseq_embeddings')

    embedding_paths = [
        # 'ESM2_alphaseq.npy',
        'ProtTrans_alphaseq.npy',
        'AbLang_alphaseq.npy',
        # 'AntiBERTy_alphaseq.npy',
        # 'BERT2DAb_features.npy'
    ]

    original_affinity = embedding_path / 'affinity.npy'
    original_affinity = np.load(str(original_affinity))

    for e in embedding_paths:
        e = embedding_path / e
        # 读取特征
        embedding = np.load(str(e))
        print(embedding.shape)
        pca = PCA(n_components=0.99)
        embedding = pca.fit_transform(X=embedding)

        mae_scores = []
        rmse_scores = []
        pearson_scores = []

        for _ in range(1):
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(range(embedding.shape[0]))):
                # print('fold{}'.format(fold_idx+1))
                model = RandomForestRegressor(n_estimators=50, n_jobs=multiprocessing.cpu_count(), max_depth=15,
                                              random_state=42)
                X_train, X_test = embedding[train_idx], embedding[test_idx]
                y_train, y_test = original_affinity[train_idx], original_affinity[test_idx]

                model.fit(X_train, y_train)
                y_pred_fold = model.predict(X_test)

                mae_scores.append(mean_absolute_error(y_test, y_pred_fold))
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred_fold)))
                pearson_scores.append(pearsonr(y_test, y_pred_fold)[0])

            mae_mean = np.mean(mae_scores)
            mae_std = np.std(mae_scores)

            rmse_mean = np.mean(rmse_scores)
            rmse_std = np.std(rmse_scores)

            pearson_mean = np.mean(pearson_scores)
            pearson_std = np.std(pearson_scores)

        # 输出结果
        print(str(e)[:-4])
        print("RMSE: {}±{}".format(round(rmse_mean, 4), round(rmse_std, 4)))
        print("MAE: {}±{}".format(round(mae_mean, 4), round(mae_std, 4)))
        print("PCC: {}±{}".format(round(pearson_mean, 4), round(pearson_std, 4)))


if __name__ == "__main__":
    # extract_from_ProtTrans()
    # extract_from_ESM2_650M()
    # extract_from_AbLang()
    # extract_from_AntiBERTy()
    # extract_from_BERT2DAb()

    compare_different_embeddings()

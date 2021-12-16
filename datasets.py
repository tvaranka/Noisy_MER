import os
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from skimage.transform import resize

def add_zeros(num):
    num = str(num)
    if len(num) == 3:
        return num
    elif len(num) == 2:
        return "0" + num
    else:
        return "00" + num

def smic(df_path="data/smic.xlsx", dataset_path="../SMIC_all_cropped/HS"):
    """Returns the dataframe containing the metadata for the videos and
       a generator object containing the videos."""
    df = pd.read_excel(df_path)
    df["n_frames"] = df["offset"] - df["onset"] + 1
    path = dataset_path
    def load_smic():
        ndf = []
        for s in os.listdir(path):
            new_path = path + "/" + s + "/micro"
            for e in os.listdir(new_path):
                video_path = new_path + "/" + e
                for v in os.listdir(video_path):
                    img = plt.imread(video_path + "/" + v + "/" + os.listdir(video_path + "/" + v)[0])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = resize(img, (170, 140))
                    n_frames = os.listdir(video_path + "/" + v).__len__()
                    video = np.zeros((img.shape[0], img.shape[1], n_frames), dtype="uint8")
                    for i, f in enumerate(os.listdir(video_path + "/" + v)):
                        img_path = video_path + "/" + v + "/" + f
                        img = plt.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img = resize(img, (170, 140))
                        img = np.round(img * 255).astype("uint8")
                        video[..., i] = img
                    ndf.append([s, e, v])
                    yield video
    return df, load_smic()

def smic_raw(df_path="data/smic.xlsx",
             dataset_path="../../Micro expressions/SMIC_all_raw/HS",
             color=False):
    """Returns the dataframe containing the metadata for the videos and
       a generator object containing the videos."""
    df = pd.read_excel(df_path)
    df["n_frames"] = df["offset"] - df["onset"] + 1
    path = dataset_path
    def load_smic():
        ndf = []
        for s in os.listdir(path):
            new_path = path + "/" + s + "/micro"
            for e in os.listdir(new_path):
                video_path = new_path + "/" + e
                for v in os.listdir(video_path):
                    img = plt.imread(video_path + "/" + v + "/" + os.listdir(video_path + "/" + v)[0])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    n_frames = os.listdir(video_path + "/" + v).__len__()
                    video = np.zeros((img.shape[0], img.shape[1], n_frames), dtype="uint8")
                    if color:
                        video = np.zeros((img.shape[0], img.shape[1], 3, n_frames), dtype="uint8")
                    for i, f in enumerate(os.listdir(video_path + "/" + v)):
                        img_path = video_path + "/" + v + "/" + f
                        img = plt.imread(img_path)
                        if not color:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        video[..., i] = img
                    ndf.append([s, e, v])
                    yield video
    return df, load_smic()

def casme(df_path="data/casme.xlsx", dataset_path="../../Micro expressions/CASME_cropped_selected"):
    #Casme with 189 samples
    df = pd.read_excel(df_path)
    df = df.drop(["Unnamed: 2", "Unnamed: 7"], axis=1)
    df = df.rename(columns={"Emotion": "emotion", "Subject": "subject", "Filename": "material",
                       "OnsetF": "onset", "ApexF1": "apex", "OffsetF": "offset"})
    df["subject"] = df["subject"].apply(lambda x: str(x) if x >= 10 else "0{}".format(x))
    df.iloc[40, 2] = 108 #Mistake in file
    df.iloc[40, 5] = 149 #Mistake in file
    df.iloc[42, 2] = 101 #Mistake in file
    df.iloc[42, 5] = 119 #Mistake in file
    df.iloc[43, 2] = 57 #Mistake in file
    df.iloc[43, 5] = 74 #Mistake in file
    df.iloc[43, 3] = 60 #missing apex
    df.iloc[54, 5] = 40
    df["n_frames"] = df["offset"] - df["onset"] + 1
    
    n_samples = df.shape[0]

    root_path = dataset_path
    
    def load_casme():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((170, 140, n_frames))
            for k, j in enumerate(range(onset, onset + n_frames)):
                if i >= 92:
                    j = add_zeros(j)
                img_path = "{}/sub{}/{}/reg_{}-{}.jpg".format(root_path, subject, material, material, j)
                img = plt.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = resize(img, (170, 140))
                video[..., k] = img
            yield video
            
    return df, load_casme()

def casme_raw(df_path="data/casme.xlsx", dataset_path="../../Micro expressions/CASME_raw_selected"):
    #Casme with 189 samples
    df = pd.read_excel(df_path)
    df = df.drop(["Unnamed: 2", "Unnamed: 7"], axis=1)
    df = df.rename(columns={"Emotion": "emotion", "Subject": "subject", "Filename": "material",
                       "OnsetF": "onset", "ApexF1": "apex", "OffsetF": "offset"})
    df["subject"] = df["subject"].apply(lambda x: str(x) if x >= 10 else "0{}".format(x))
    df.iloc[40, 2] = 108 #Mistake in file
    df.iloc[40, 5] = 149 #Mistake in file
    df.iloc[42, 2] = 101 #Mistake in file
    df.iloc[42, 5] = 119 #Mistake in file
    df.iloc[43, 2] = 57 #Mistake in file
    df.iloc[43, 5] = 74 #Mistake in file
    df.iloc[54, 5] = 40
    df["n_frames"] = df["offset"] - df["onset"] + 1
    
    n_samples = df.shape[0]

    root_path = dataset_path
    
    def load_casme():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((170, 140, n_frames), dtype="uint8")
            for k, j in enumerate(range(onset, onset + n_frames)):
                if i >= 92:
                    j = add_zeros(j)
                img_path = "{}/sub{}/{}/{}-{}.jpg".format(root_path, subject, material, material, j)
                img = plt.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = resize(img, (170, 140))
                video[..., k] = img
            yield video
            
    return df, load_casme()

def casme2(df_path="data/CASME2-coding-updated.xlsx", dataset_path="../CASME2_Cropped/Cropped"):
    #Casme 2 path with 256/247 samples, length(fear + sadness) = 9, removed
    df = pd.read_excel(df_path)
    df = df.drop(["Unnamed: 2", "Unnamed: 6"], axis=1)
    df = df.rename(columns={"Estimated Emotion": "emotion", "Subject": "subject", "Filename": "material",
                       "OnsetFrame": "onset", "ApexFrame": "apex", "OffsetFrame": "offset"})
    df.iloc[60, 4] = 91 #Mistake in file, change offset to 91
    df["n_frames"] = df["offset"] - df["onset"] + 1
    df["subject"] = df["subject"].apply(lambda x: str(x) if x >= 10 else "0{}".format(x))
    df = df[~df["emotion"].isin(["fear", "sadness"])]
    df = df.reset_index()

    # missing apex, changed based on looking at OF
    df.iloc[45, 4] = 81
    df.iloc[29, 4] = 279
    df.iloc[35, 4] = 68
    df.iloc[43, 4] = 77
    df.iloc[51, 4] = 166
    df.iloc[53, 4] = 100
    df.iloc[60, 4] = 78
    df.iloc[115, 4] = 187
    df.iloc[116, 4] = 89
    df.iloc[124, 4] = 80
    df.iloc[134, 4] = 88
    df.iloc[145, 4] = 134
    df.iloc[153, 4] = 231
    df.iloc[166, 4] = 53
    df.iloc[173, 4] = 111
    df.iloc[197, 4] = 91
    df.iloc[198, 4] = 103
    df.iloc[226, 4] = 98
    df.iloc[229, 4] = 153
    df.iloc[230, 4] = 98
    
    n_samples = df.shape[0]

    root_path = dataset_path
    
    def load_casme2():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((170, 140, n_frames), dtype="uint8")
            for k, j in enumerate(range(onset, onset + n_frames)):
                img_path = "{}/sub{}/{}/reg_img{}.jpg".format(root_path, subject, material, j)
                img = plt.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = resize(img, (170, 140))
                img = np.round(img * 255).astype("uint8")
                video[..., k] = img
            yield video
            
    return df, load_casme2()


def casme2_raw(df_path="data/CASME2-coding-updated.xlsx",
               dataset_path="../../Micro expressions/CASME2_RAW_selected/CASME2_RAW_selected",
               color=False):
    #Casme 2 path with 256/247 samples, length(fear + sadness) = 9, removed
    df = pd.read_excel(df_path)
    df = df.drop(["Unnamed: 2", "Unnamed: 6"], axis=1)
    df = df.rename(columns={"Estimated Emotion": "emotion", "Subject": "subject", "Filename": "material",
                       "OnsetFrame": "onset", "ApexFrame": "apex", "OffsetFrame": "offset"})
    df.iloc[60, 4] = 91 #Mistake in file, change offset to 91
    df["n_frames"] = df["offset"] - df["onset"] + 1
    df["subject"] = df["subject"].apply(lambda x: str(x) if x >= 10 else "0{}".format(x))
    df = df[~df["emotion"].isin(["fear", "sadness"])]
    df = df.reset_index()
    # missing apex, changed based on looking at OF
    df.iloc[45, 4] = 81
    df.iloc[29, 4] = 279
    df.iloc[35, 4] = 68
    df.iloc[43, 4] = 77
    df.iloc[51, 4] = 166
    df.iloc[53, 4] = 100
    df.iloc[60, 4] = 78
    df.iloc[115, 4] = 187
    df.iloc[116, 4] = 89
    df.iloc[124, 4] = 80
    df.iloc[134, 4] = 88
    df.iloc[145, 4] = 134
    df.iloc[153, 4] = 231
    df.iloc[166, 4] = 53
    df.iloc[173, 4] = 111
    df.iloc[197, 4] = 91
    df.iloc[198, 4] = 103
    df.iloc[226, 4] = 98
    df.iloc[229, 4] = 153
    df.iloc[230, 4] = 98
    
    n_samples = df.shape[0]

    root_path = dataset_path
    
    def load_casme2():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]

            video = np.zeros((480, 640, n_frames), dtype="uint8")
            if color:
                video = np.zeros((480, 640, 3, n_frames), dtype="uint8")
            for k, j in enumerate(range(onset, onset + n_frames)):
                img_path = "{}/sub{}/{}/img{}.jpg".format(root_path, subject, material, j)
                img = plt.imread(img_path)
                if not color:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                video[..., k] = img
            yield video
            
    return df, load_casme2()

def samm(df_path="data/SAMM_Micro_FACS_Codes_v2.xlsx", dataset_path="../SAMM_CROP"):
    #SAMM with 159 samples
    df = pd.read_excel(df_path)
    #preprocess the dataframe as it contains some text
    cols = df.loc[12].tolist()
    data = df.iloc[13:].reset_index()
    new_cols = {df.columns.tolist()[i]: cols[i] for i in range(len(cols))}
    df = pd.DataFrame(data).rename(columns=new_cols)
    df = df.rename(columns={"Estimated Emotion": "emotion", "Subject": "subject", "Filename": "material",
                       "Onset Frame": "onset", "Apex Frame": "apex", "Offset Frame": "offset"})
    df.iloc[56, 6] = 5793 #mistake in file
    df.loc[125, "apex"] = 1105 #mistake in file, set arbitrarily
    df.loc[132, "apex"] = 4945
    df.loc[133, "apex"] = 5188
    df["n_frames"] = df["offset"] - df["onset"] + 1
    root_path = dataset_path
    n_samples = df.shape[0]
    
    def load_samm():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]
            
            files = list(os.walk("{}/{}/{}".format(root_path, subject, material)))[0][2]
            video = np.zeros((170, 140, n_frames), dtype="uint8")
            for k, f in enumerate(files):
                img_path = "{}/{}/{}/{}".format(root_path, subject, material, f)
                img = plt.imread(img_path)
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = resize(img, (170, 140))
                img = np.round(img * 255).astype("uint8")
                video[..., k] = img
            yield video
            
    return df, load_samm()

def samm_raw(df_path="data/SAMM_Micro_FACS_Codes_v2.xlsx",
             dataset_path="../../Micro expressions/SAMM",
             color=False):
    #SAMM with 159 samples
    df = pd.read_excel(df_path)
    #preprocess the dataframe as it contains some text
    cols = df.loc[12].tolist()
    data = df.iloc[13:].reset_index()
    new_cols = {df.columns.tolist()[i]: cols[i] for i in range(len(cols))}
    df = pd.DataFrame(data).rename(columns=new_cols)

    df = df.rename(columns={"Estimated Emotion": "emotion", "Subject": "subject", "Filename": "material",
                       "Onset Frame": "onset", "Apex Frame": "apex", "Offset Frame": "offset"})
    df.iloc[56, 6] = 5793 #mistake in file
    df.loc[125, "apex"] = 1105 #mistake in file, set arbitrarily
    df.loc[132, "apex"] = 4945
    df.loc[133, "apex"] = 5188
    df["n_frames"] = df["offset"] - df["onset"] + 1
    root_path = dataset_path
    n_samples = df.shape[0]
    
    def load_samm():
        for i in range(n_samples):
            subject = df["subject"][i]
            material = df["material"][i]
            emotion = df["emotion"][i]
            onset = df["onset"][i]
            n_frames = df["n_frames"][i]
            
            files = list(os.walk("{}/{}/{}".format(root_path, subject, material)))[0][2]
            video = np.zeros((650, 960, n_frames), dtype="uint8")
            if color:
                video = np.zeros((650, 960, 3, n_frames), dtype="uint8")
            for k, f in enumerate(files):
                img_path = "{}/{}/{}/{}".format(root_path, subject, material, f)
                img = plt.imread(img_path)
                if color:
                    img = np.array([img, img, img]).transpose(1, 2, 0)
                video[..., k] = img
            yield video
            
    return df, load_samm()

def megc(dataset="cropped", color=False):
    if dataset == "raw":
        df_samm, load_samm = samm_raw(color=color)
        df_casme2, load_casme2 = casme2_raw(color=color)
        df_smic, load_smic = smic_raw(color=color)
    else:
        df_samm, load_samm = samm()
        df_casme2, load_casme2 = casme2()
        df_smic, load_smic = smic()

    #Remove "others" from casme2
    indices = df_casme2[df_casme2["emotion"] != "others"]["emotion"].index.tolist()
    load_casme2 = (video for i, video in enumerate(load_casme2) if i in indices)
    df_casme2 = df_casme2[df_casme2["emotion"] != "others"].copy()
    #Set the correct emotions
    df_casme2.loc[df_casme2["emotion"].isin(["disgust", "repression"]), "emotion"] = "negative"
    df_casme2.loc[df_casme2["emotion"] == "happiness", "emotion"] = "positive"
    
    #remove "others" from samm
    indices2 = df_samm[df_samm["emotion"] != "Other"]["emotion"].index.tolist()
    load_samm = (video for i, video in enumerate(load_samm) if i in indices2)
    df_samm = df_samm[df_samm["emotion"] != "Other"].copy()
    #Set the correct emotions
    df_samm.loc[df_samm["emotion"].isin(
        ["Anger", "Contempt", "Disgust", "Sadness", "Fear"]), "emotion"] = "negative"
    df_samm.loc[df_samm["emotion"] == "Happiness", "emotion"] = "positive"
    df_samm.loc[df_samm["emotion"] == "Surprise", "emotion"] = "surprise"

    #merge dataframes and iterators
    df_smic = df_smic.rename(columns={"Unnamed: 0": "index"})
    df = pd.concat([df_casme2, df_smic, df_samm], sort=True)
    df = df.reset_index()
    df = df.drop(["Duration", "Inducement Code", "Micro", "Notes", "Objective Classes"], axis=1)
    
    #add column for dataset information
    df["dataset"] = "casme2"
    df.loc[148:312, "dataset"] = "smic"
    df.loc[312:, "dataset"] = "samm"

    #apex is the apex frame based on file number and apexf for image number in sequence
    df.loc[df.index[df["apex"].isnull()], "apex"] = (df[df["apex"].isnull()]["offset"] - df[df["apex"].isnull()]["onset"]) / 2
    df["apexf"] = df["apex"] - df["onset"]
    df.loc[df["dataset"] == "smic", "apexf"] = df.loc[df["dataset"] == "smic", "apex"]
    df.loc[:, "apex"] = round(df["apex"].astype("float")).astype("int")
    df.loc[:, "apexf"] = round(df["apexf"].astype("float")).astype("int")

    df = df.drop([17, 26, 130]).drop("level_0", axis=1).reset_index()
    
    load_data = chain(load_casme2, load_smic, load_samm)
    #remove 17, 26, 130
    load_data = (video for i, video in enumerate(load_data) if i not in [17, 26, 130])

    return df, load_data
    















    
    

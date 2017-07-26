import numpy as np
import tensorflow as tf


##############################################################################
'''DataFrame Padding Function'''
##############################################################################

def padding(df):
    df_split = np.array([ [df[m][n].split() for n in range(len(df[m]))] for m in range(len(df))])
    lens_words = np.array([ [len(df_split[m][n]) for n in range(len(df_split[m]))] for m in range(len(df_split))])
    lens_sen = np.array([len(df_split[m]) for m in range(len(df_split))])
    max_words = max([max(lens_words[m]) for m in range(len(df_split))])
    max_sen = max(lens_sen)
    df_split_pad = np.array( [ [df_split[m][n] + ["#PAD"]*(max_words - lens_words[m][n]) for n in range(len(df_split[m]))]
                            for m in range(len(df_split))])
    df_split_pad = np.array( [ df_split_pad[m] + [["#PAD"]*max_words]*(max_sen - lens_sen[m]) for m in range(len(df_split_pad))])
    df_pad = np.array( [ [[" ".join(df_split_pad[m][n])] for n in range(len(df_split_pad[m]))] for m in range(len(df_split_pad)) ])
    return df_pad

def padding_split(df):
    df_split = np.array([ [df[m][n].split() for n in range(len(df[m]))] for m in range(len(df))])
    lens_words = np.array([ [len(df_split[m][n]) for n in range(len(df_split[m]))] for m in range(len(df_split))])
    lens_sen = np.array([len(df_split[m]) for m in range(len(df_split))])
    max_words = max([max(lens_words[m]) for m in range(len(df_split))])
    max_sen = max(lens_sen)
    df_split_pad = np.array( [ [df_split[m][n] + ["#PAD"]*(max_words - lens_words[m][n]) for n in range(len(df_split[m]))]
                            for m in range(len(df_split))])
    df_split_pad = np.array( [ df_split_pad[m] + [["#PAD"]*max_words]*(max_sen - lens_sen[m]) for m in range(len(df_split_pad))])
    #df_pad = np.array( [ [[" ".join(df_split_pad[m][n])] for n in range(len(df_split_pad[m]))] for m in range(len(df_split_pad)) ])
    return df_split_pad



##############################################################################
'''Padding batches of sets of vectors'''
##############################################################################
def padding_vec(df, vec_dim):
    lens = np.array([len(batch) for batch in df])
    lens_max = max(lens)
    pad = [tf.zeros(vec_dim)]
    df = [list(elem) + pad*(lens_max - lens[i])
                  for i,elem in enumerate(df)]
    return df

def padding_scalar(df):
    lens = np.array([len(batch) for batch in df])
    lens_max = max(lens)
    pad = [-1]
    df = np.array([list(elem) + pad*(lens_max - lens[i])
                  for i,elem in enumerate(df)])
    return df


##############################################################################
'''Compare two sets and pad the smaller to match the bigger'''
##############################################################################
def padding_compare(df1, df2):
    len1 = df1.shape[1]
    len2 = df2.shape[1]
    vec_dim = df1.shape[2]
    if len1 > len2:
        pad = np.split(np.zeros(vec_dim *(len1-len2)), (len1-len2))
        df2 = np.array([ list(elem) + list(pad) for elem in df2])
        return df1, df2
    elif len1 < len2:
        pad = np.split(np.zeros(vec_dim *(len2-len1)), (len2-len1))
        df1 = np.array([ list(elem) + list(pad) for elem in df1])
        return df1, df2
    else:
        return df1, df2

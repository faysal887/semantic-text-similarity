import numpy as np
import faiss
from faiss import normalize_L2
import pandas as pd
import sys, os
import os.path


def apply_threshold(row, k=5, threshold=0.95):
    try:
        row=row.to_frame().T
        row.iloc[0,:k]=row.iloc[0,:k].astype(float)
        r=row.iloc[0, :k]
        ind = (r<threshold).values.flatten().tolist()
        ind += ind
        ind2 = [False]*len(row.iloc[0, k+k:])
        ind += ind2
        row.iloc[:, ind] = np.nan
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname =os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)
    return row
    
    
def search_faiss(index_faiss, vec, k=5):
    top_k = index_faiss.search(vec, k)
    return list(top_k)


def create_index(df, ids_col, xb_faiss_col):
    try:
        ids=np.array(df[ids_col].str.slice(start=3).tolist()).astype(int)
        xb_faiss = np.float32(np.asarray(df[xb_faiss_col].tolist()))

        d  = xb_faiss.shape[1] 

        normalize_L2(xb_faiss)

        index_faiss = faiss.IndexIDMap(faiss.IndexFlatIP(d))
        index_faiss.add_with_ids(xb_faiss, ids)

#         print("FAISS index trained: ", index_faiss.is_trained)
#         print("Total db vectors: ", index_faiss.ntotal)

        return index_faiss
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname =os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Exception get_similar_tickets: ", e, exc_type, fname, exc_tb.tb_lineno)
        return False
    

def add_neighbours_text(mydf, train_df, k):
    try:
        for i in range(1,k+1):
            vals=mydf["NN"+str(i)+"_number"].tolist()
            if pd.isnull(vals).all():
                mydf["NN"+str(i)+"_text"]=np.nan
            else:
                for exp_id in mydf["NN"+str(i)+"_number"].tolist():
                    if pd.isnull(exp_id):
                        mydf.loc[mydf["NN"+str(i)+"_number"]==exp_id, "NN"+str(i)+"_text"] = np.nan
                    else:
                        mydf.loc[mydf["NN"+str(i)+"_number"]==exp_id, "NN"+str(i)+"_text"] = train_df.set_index("key").loc[exp_id]['text']
        return mydf
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname =os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Exception add_neighbours_text: ", e, exc_type, fname, exc_tb.tb_lineno)
        return False
    

def get_similar_tickets(train_df, test_df, ids_col, xb_faiss_col, k=5, threshold=0.95):
#     print("starting operations")
    
    try:
#         print("creating faiss index")
        index_faiss = create_index(train_df, ids_col, xb_faiss_col)
        
#         print("starting calculations")
        if "emb" not in test_df.columns: 
            print("embeddings not found")
            return False

        xq = np.float32(np.asarray(test_df["emb"].tolist()))
        normalize_L2(xq)
#         print("query vector created")
        
        result=search_faiss(index_faiss, xq, k)
#         print("got results from FAISS")
        
        nums = lambda t: "INC"+str(int(t))
        vfunc = np.vectorize(nums)
        result[1]=vfunc(result[1])

        mydf=pd.DataFrame(np.column_stack(result))
        mydf["query_key"]=test_df["key"].tolist()
        mydf["query_text"]=test_df["text"].tolist()
#         print("creating result dataframe")

        mydf=mydf.rename(columns = dict(zip(range(0,k), [f'NN{n+1}_score' for n in range(0,k)])))
        mydf=mydf.rename(columns = dict(zip(range(k, (k*2)), [f'NN{n+1}_number' for n in range(0,k)])))


        res=mydf.apply(apply_threshold, axis=1, args=(k,threshold))
        res=pd.concat(res.values.tolist())

        res = res.assign(is_similar=np.where(pd.isnull(res["NN1_score"]) == True , False, True))

        res = pd.merge(test_df, res.rename(columns={"query_key": "key"}), on="key", how="inner")
        res.rename(columns={"query_key": "key"}, inplace=True)
#         res = add_neighbours_text(res, train_df, k)
        
        return res
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname =os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Exception get_similar_tickets: ", e, exc_type, fname, exc_tb.tb_lineno)
        return False
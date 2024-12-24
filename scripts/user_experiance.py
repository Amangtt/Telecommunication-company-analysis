import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import streamlit as st
class experiance:
    
    #Agregation of datas needed to understand users experiance 
    def per_cus(df):
        df['total_TCP']= df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']
        df['total_RTT']= df['Avg RTT UL (ms)'] + df['Avg RTT DL (ms)']
        df['total_TP']= df['Avg Bearer TP UL (kbps)'] + df['Avg Bearer TP UL (kbps)']
            
        df= df.groupby('MSISDN/Number').agg(
            AVG_TCP=('total_TCP','mean'),
            AVG_RTT=('total_RTT','mean'),
            AVG_TP=('total_TP','mean'),
            )
        return df
    #Lists 10 of the top, bottom and frequent values of a column and you pass the name of the column you want calcuated
    def list(df, col):
        
        top= df[col].nlargest(10).reset_index(drop=True)
        top.columns = ['Index', col]
        bottom= df[col].nsmallest(10).reset_index(drop=True)
        bottom.columns = ['Index', col]
        frequent= df[col].value_counts().nlargest(10).reset_index(drop=True)
        frequent.columns = ['Index', col]

        return top,bottom,frequent
    

    #Displays total Throughput per handset type
    def per_handset_TP(df):
        df['total_TP']= df['Avg Bearer TP UL (kbps)'] + df['Avg Bearer TP UL (kbps)']
        summary= df.groupby('Handset Type')['total_TP'].mean().reset_index()
        summary=summary.sort_values(by='total_TP',ascending=False)
        top_5 = summary.head(5)
        plt.figure(figsize=(10, 6))
        plt.bar(top_5['Handset Type'], top_5['total_TP'], color=['blue', 'orange', 'green'])
        plt.title('Average Throughput per Handset Type')
        plt.xlabel('Handset Type')
        plt.ylabel('Average Throughput')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        return summary
            
     #Displays total TCP per handset type
    def per_handset_TCP(df):
        df['total_TCP']= df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']
        summary= df.groupby('Handset Type')['total_TCP'].mean().reset_index()
        summary=summary.sort_values(by='total_TCP',ascending=False)
        top_5=summary.head(5)
        plt.figure(figsize=(10, 6))
        plt.bar(top_5['Handset Type'], top_5['total_TCP'], color=['blue', 'orange', 'green'])
        plt.title('Average TCP per Handset Type')
        plt.xlabel('Handset Type')
        plt.ylabel('Average TCP')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        return summary
            
    #Creates a cluster based on users experiance
    def experiance(ad_data):
            
        features=ad_data[['AVG_TCP','AVG_RTT','AVG_TP']]
        scalar=StandardScaler()
        normalized= scalar.fit_transform(features)
        kmean=KMeans(n_clusters=3, random_state=42)
        cluster=kmean.fit_predict(normalized)
        ad_data['Cluster']= cluster
        cluster_counts = ad_data['Cluster'].value_counts()
        return ad_data
    
    #Provids avarage value per cluster
    def cluss_avg(ad_data):
            
        cluster_avg=ad_data.groupby('Cluster').agg(
                {'AVG_TCP': 'mean',
                'AVG_RTT': 'mean',
                'AVG_TP': 'mean'}
            )
        return cluster_avg

    
 
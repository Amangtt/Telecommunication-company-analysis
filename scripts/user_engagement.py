import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class engagement:
    def calculate_engagement_metrics(df):
       
        # Group by 'MSISDN/Number' and aggregate metrics
        engagement_metrics = df.groupby('MSISDN/Number').agg(
            session_frequency=('Bearer Id', 'count'),  # Count of sessions
            session_duration=('Dur. (ms)', 'sum'),     # Total duration of all sessions in ms
            total_download_traffic=('Total DL (Bytes)', 'sum'),  # Total download traffic
            total_upload_traffic=('Total UL (Bytes)', 'sum')      # Total upload traffic
        ).reset_index()

       

        # Calculate total traffic
        engagement_metrics['total_traffic'] = engagement_metrics['total_download_traffic'] + engagement_metrics['total_upload_traffic']

        # Get top 10 metrics
        top_10_sessions = engagement_metrics.nlargest(10, 'session_frequency').reset_index()
        top_10_duration = engagement_metrics.nlargest(10, 'session_duration').reset_index()
        top_10_traffic = engagement_metrics.nlargest(10, 'total_traffic').reset_index()

        # Display results
        session10 = top_10_sessions[['MSISDN/Number', 'session_frequency']]
        dur10 = top_10_duration[['MSISDN/Number', 'session_duration']]
        traffic10 = top_10_traffic[['MSISDN/Number', 'total_traffic']]

        return engagement_metrics, top_10_sessions, top_10_duration, top_10_traffic

    # This fnction is used to normalize and classfiy the data into 3 cluster
    def normalize(metrics):
        met=pd.DataFrame(metrics)
        data_to_normalize = met[['session_frequency', 'session_duration', 'total_traffic']]
        scaler= MinMaxScaler()
        normalized= scaler.fit_transform(data_to_normalize)
        kmean= KMeans(n_clusters=3, random_state=42)
        met['Cluster'] = kmean.fit_predict(normalized)
        cluster_counts = met['Cluster'].value_counts()
        print(cluster_counts)
        return met
    
    # This Function evaluates each cluster
    def cluster_eval(clusters):
        cluster_summary = clusters.groupby('Cluster').agg(
            min_session_frequency=('session_frequency', 'min'),
            max_session_frequency=('session_frequency', 'max'),
            avg_session_frequency=('session_frequency', 'mean'),
            
            min_session_duration=('session_duration', 'min'),
            max_session_duration=('session_duration', 'max'),
            avg_session_duration=('session_duration', 'mean'),
            
            min_total_traffic=('total_traffic', 'min'),
            max_total_traffic=('total_traffic', 'max'),
            avg_total_traffic=('total_traffic', 'mean'),
        
        ).reset_index()
        layout_rows = 4  
        layout_cols = 3  

        cluster_summary.plot(
            x='Cluster', 
            kind='bar', 
            figsize=(12, 12), 
            subplots=True, 
            layout=(layout_rows, layout_cols), 
            legend=False
        )
        plt.suptitle('Non-Normalized Metrics by Cluster', fontsize=16)
        plt.xlabel('Cluster')
        plt.ylabel('Metrics Value')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    # This Function is used to identify the top 10 users for each application
    def top10_users(df):
        cols_to_sum = [
            'Total_DLUL_netflix',
            'Total_DLUL_Youtube',
            'Total_DLUL_Gaming',
            'Total_DLUL_social',
            'Total_DLUL_email',
            'Total_DLUL_Other',
            'Total_DLUL_google'
        ]

        top_10_dict = {}

        for col in cols_to_sum:
           
            total_traffic = df.groupby('MSISDN/Number')[col].sum().reset_index()
            top_10 = total_traffic.nlargest(10, col)
            
            top_10_dict[col] = top_10

        return top_10_dict
    # This function is used to specify the top 3 most used applications
    def most_used(df):
        cols = [
        'Total_DLUL_netflix',
        'Total_DLUL_Youtube',
        'Total_DLUL_Gaming',
        'Total_DLUL_social',
        'Total_DLUL_email',
        'Total_DLUL_Other',
        'Total_DLUL_google'
        ]
        total= df[cols].sum()
        top3= total.nlargest(3)
        
        plt.figure(figsize=(8, 8))
        plt.pie(top3, labels=top3.index, autopct='%1.1f%%', startangle=140)
        plt.title('Top 3 Most Used Applications')
        plt.axis('equal')  
        plt.show()
        return top3
    # This function is used to calculate the optimal number of cluster for kmeans
    def km(metrics):
        met=pd.DataFrame(metrics)
        data = met[['session_frequency', 'session_duration', 'total_traffic']]
        k_range= range(1,11)
        wcss=[]
        for k in k_range:
            kmean=KMeans(n_clusters=k, random_state=42)
            kmean.fit_transform(data)
            wcss.append(kmean.inertia_)
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, wcss, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('WCSS')
        plt.xticks(k_range)
        plt.grid()
        plt.show()
    
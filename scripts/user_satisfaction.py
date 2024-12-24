import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

class saisfaction:    
    #Calculating engagement score
    def calculate_engagement_scores(user_data,avg):
        user_data['Engagement Score'] = np.nan  # Initialize column for scores
        centroids = avg
        features=['session_frequency', 'session_duration', 'total_traffic']
        cen_fea=['avg_session_frequency','avg_session_duration','avg_total_traffic']
        scaler= MinMaxScaler()
        user_data[['session_frequency', 'session_duration', 'total_traffic']]= scaler.fit_transform(user_data[features])
        centroids[['avg_session_frequency','avg_session_duration','avg_total_traffic']]=scaler.fit_transform(centroids[cen_fea])
        for cluster_id in centroids.index:
            
            centroid=centroids.loc[cluster_id]
            
            cluster_users = user_data[user_data['Cluster'] == cluster_id]
            
            if not cluster_users.empty:
                distances = np.linalg.norm(cluster_users[['session_frequency', 'session_duration', 'total_traffic']] - centroid.values, axis=1)
                user_data.loc[user_data['Cluster'] == cluster_id, 'Engagement Score'] = distances
        sorted_data=user_data.sort_values(by= 'Engagement Score',ascending=False )
        return sorted_data
    
    #Calculating experinace score
    def calculate_experiance_scores(user_data,avg_clus):
        user_data['Experiance Score'] = np.nan  # Initialize column for scores
        centroids = avg_clus
        features=['AVG_TCP', 'AVG_RTT', 'AVG_TP']
        cen_fea=['AVG_TCP', 'AVG_RTT', 'AVG_TP']
        scaler= MinMaxScaler()
        user_data[['AVG_TCP', 'AVG_RTT', 'AVG_TP']]= scaler.fit_transform(user_data[features])
        centroids[['AVG_TCP', 'AVG_RTT', 'AVG_TP']]=scaler.fit_transform(centroids[cen_fea])
        for cluster_id in centroids.index:
            
            centroid=centroids.loc[cluster_id]
            
            cluster_users = user_data[user_data['Cluster'] == cluster_id]
            
            if not cluster_users.empty:
                distances = np.linalg.norm(cluster_users[['AVG_TCP', 'AVG_RTT', 'AVG_TP']] - centroid.values, axis=1)
                user_data.loc[user_data['Cluster'] == cluster_id, 'Experiance Score'] = distances
        sorted_data=user_data.sort_values(by= 'Experiance Score',ascending=False ).reset_index()
        return sorted_data
  
    #Calculating satisfaction score
    def calculate_satisfaction_score(engagements,experiances):
        engagement_score=engagements[['MSISDN/Number','Engagement Score']]
        experiance_score=experiances[['MSISDN/Number','Experiance Score']]
        merged_df = pd.merge(engagements, experiances, on='MSISDN/Number')
        
        merged_df['Satisfaction Score'] = (merged_df['Engagement Score'] + merged_df['Experiance Score']) / 2
        top_satisfied_customers = merged_df.sort_values(by='Satisfaction Score', ascending=False).head(10)
        return merged_df
    #Creating a linear regression model and testing it 
    def model(satis):
        x=satis[['Engagement Score','Experiance Score']]
        y=satis['Satisfaction Score']
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)
        model=LinearRegression()
        model.fit(x_train,y_train)
        yhat=model.predict(x_test)
        print(f"Testset Mean Absolute Error: {mean_absolute_error(y_test, yhat)}")
        print(f"Testset Mean Squared Error: {mean_squared_error(y_test, yhat)}")
        print(f"Testset R² Score: {r2_score(y_test, yhat)}")
   
    #Clustering into satisfied and unsatisfied
    def cluster_experiance_engagement(satis):
        dataset=satis[['MSISDN/Number','Engagement Score','Experiance Score','Satisfaction Score']]
        kmean=KMeans(n_clusters=2,random_state=55)
        dataset['Cluster']=kmean.fit_predict(dataset[['Engagement Score','Experiance Score']])
        cluster_counts = dataset['Cluster'].value_counts()
        
        return dataset,cluster_counts
    
    def avg(dataset):
        cluster_summary=dataset.groupby('Cluster').agg({
            'Satisfaction Score':'mean',
            'Experiance Score':'sum',
            'Engagement Score':'sum'
        }).reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_summary.plot(x='Cluster', kind='bar', ax=ax)

        # Customize the plot
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Values')
        ax.set_title('Cluster Averages and Sums')
        ax.legend(['Satisfaction Score', 'Experience Score', 'Engagement Score'])
        plt.xticks(rotation=45) 

        plt.show()
        return cluster_summary
    






import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Reading CSV files and preprocessing
column_name = ['ID', 'Entity', 'Sentiment', 'Content']
df = pd.read_csv(r"C:\Users\ASUS\Downloads\archive (5)\twitter_training.csv", names=column_name)
df2 = pd.read_csv(r"C:\Users\ASUS\Downloads\archive (5)\twitter_validation.csv", names=column_name)
df = df.drop_duplicates(subset='ID').reset_index(drop=True)

# Overall Sentiment Distribution
sentiment = df['Sentiment'].value_counts()
plt.figure(figsize= (12,14))

# Plotting overall sentiment distribution
plt.subplot(2,2,1)
bars = plt.bar(sentiment.index.astype(str), sentiment.values, color='green')

# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Overall Sentiment Distribution')

# Sentiment Distribution for Facebook
plt.subplot(2,2,2)
fb_sentiment = df2[df2['Entity'] == 'Facebook']['Sentiment']
fb_sentiment_count = fb_sentiment.value_counts()
plt.pie(fb_sentiment_count, labels=fb_sentiment_count.index, autopct='%1.1f%%', startangle=90)
plt.title('Sentiment Distribution for Facebook')

# Sentiment Distribution by Entity (Bar Plot)
plt.subplot(2,2,3)
group = df[df['Sentiment'].isin(['Negative', 'Neutral', 'Positive'])]
group = group.groupby(['Entity', 'Sentiment']).size().reset_index(name='count')
piv_group = group.pivot(index='Entity', columns='Sentiment', values='count')
piv_group = piv_group.iloc[:15]  # Taking first 15 entities
width = 0.2
# Plotting bars for each sentiment
plt.bar(range(len(piv_group)), piv_group['Positive'], width, label='Positive', color='orange')
plt.bar([i + width for i in range(len(piv_group))], piv_group['Neutral'], width, label='Neutral', color='blue')
plt.bar([i + (width * 2) for i in range(len(piv_group))], piv_group['Negative'], width, label='Negative', color='green')
plt.xticks([i + width for i in range(len(piv_group))], piv_group.index, rotation=90)
plt.xlabel('Entity')
plt.ylabel('Count')
plt.title('Sentiment Distribution Among Companies')
plt.legend()

# Positive Sentiment Distribution (Horizontal Bar Plot)
plt.subplot(2,2,4)
positive = df2[df2['Sentiment'] == 'Positive']['Entity']
positive_count = positive.value_counts()
positive_count = positive_count.iloc[:-15]
print(positive_count.size)
plt.barh(positive_count.index, positive_count.values, color='purple')
plt.xlabel('Count')
plt.ylabel('Entity')
plt.title('Positive Sentiment Distribution')
plt.tick_params(axis='y', pad=10)  # Increase spacing between y-axis labels and bars
plt.tick_params(axis='x', pad=10)

# Adjust layout with spacing
plt.subplots_adjust(wspace=0.4, hspace=.4)  # Adjust horizontal and vertical spacing

plt.tight_layout()
plt.show()

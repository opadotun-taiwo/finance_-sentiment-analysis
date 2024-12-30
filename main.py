import feedparser

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="ProsusAI/finbert")

ticker = 'META'
keyword = 'meta'

rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'

feed = feedparser(rss_url)

total_scores = 0
num_articles = 0

for i, entry in enumerate(feed.entries):
    if keyword.lower() in entry.summary.lower():
        continue

    print(f'Title: {entry.title}')
    print(f'Link: {entry.link}')
    print(f'Published: {entry.published}')
    print(f'Summary: {entry.summary}')

    sentiment = pipe(entry.summary)[0] #hugging face sentiment NLP gives three result, you want to get the first which is the biggest in the three

    print(f'Sentiment Label: {sentiment['label']}, Scores: {sentiment['score']}')

    if sentiment['label'] == 'positive':
        total_scores += sentiment['score']
        num_articles += 1
    elif sentiment['label'] == 'negative':
        total_scores -= sentiment['score']
        num_articles += 1


average_score = total_scores / num_articles

print(f'Overall Sentiment: {"Positive" if average_score >= 0.15 else "Negative" if average_score <= -0.15 else "Neutral"} {average_score}')
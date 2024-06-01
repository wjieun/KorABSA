import pandas as pd

def print_acc(df, on):
    global test_df, absa_df
    both = df[df['_merge'] == 'both']

    print(on)
    print(f'{len(both)}, recall: {len(both) / len(test_df):.2f}, precision: {len(both) / len(absa_df):.2f}')
    print()


# Our
test_df = pd.read_csv('Dataset/Sentiment/cosmetic_Sentiment_test.csv')
absa_df = pd.read_csv('Results/ABSA.csv')

test_df['Text'] = test_df['Text'].apply(lambda x: str(x).replace(' ', ''))
absa_df['Text'] = absa_df['Text'].apply(lambda x: str(x).replace(' ', ''))
result = pd.merge(test_df, absa_df, on=['Index', 'Text'], how='outer', indicator=True)
print_acc(result, 'Index, Text')

result = pd.merge(test_df, absa_df, on=['Index', 'Text', 'Aspect'], how='outer', indicator=True)
print_acc(result, 'Index, Text, Aspect')

result = pd.merge(test_df, absa_df, on=['Index', 'Text', 'Sentiment'], how='outer', indicator=True)
print_acc(result, 'Index, Text, Sentiment')

result = pd.merge(test_df, absa_df, on=['Index', 'Text', 'Aspect', 'Sentiment'], how='outer', indicator=True)
print_acc(result, 'Index, Text, Aspect, Sentiment')
result.to_csv('Results/compare_result_our.csv', encoding='UTF-8', index=False)


# GPT
test_df = pd.read_csv('Dataset/Sentiment/cosmetic_Sentiment_test.csv')
absa_df = pd.read_csv('Results/gpt_ABSA.csv')

test_df['Text'] = test_df['Text'].apply(lambda x: str(x).replace(' ', ''))
absa_df['Text'] = absa_df['Text'].apply(lambda x: str(x).replace(' ', ''))
result = pd.merge(test_df, absa_df, on=['Index', 'Text'], how='outer', indicator=True)
print_acc(result, 'Index, Text')

result = pd.merge(test_df, absa_df, on=['Index', 'Text', 'Aspect'], how='outer', indicator=True)
print_acc(result, 'Index, Text, Aspect')

result = pd.merge(test_df, absa_df, on=['Index', 'Text', 'Sentiment'], how='outer', indicator=True)
print_acc(result, 'Index, Text, Sentiment')

result = pd.merge(test_df, absa_df, on=['Index', 'Text', 'Aspect', 'Sentiment'], how='outer', indicator=True)
print_acc(result, 'Index, Text, Aspect, Sentiment')
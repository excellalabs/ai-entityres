from collections import Counter
import csv
import pickle

## Get the data
with open('./data/data-00000-of-00010','rb') as file:
    data = file.read().decode('utf-8')
    data_blocks = data.split('URL')
    data_len = len(data_blocks)
    del data
    
    ## Initialize the data lists
    df = []
    for i,block in enumerate(data_blocks):
        if len(block) <= 1:
            continue
        
        mentions = []
        tokens = []
        ## Split data by line
        for line in block.split('\n'):
            if 'MENTION\t' in line:
                mentions.append(line.split('\t'))
            if 'TOKEN\t' in line:
                tokens.append(line.split('\t'))
        
        ## Assemble TF/IDF keywords with mention and associated URL
        for mention in mentions:
            if len(mention) < 4: ## Handle some blank mentions
                continue
            context = []
            for token in tokens:
                if len(token) < 3:  ##Handle some blank tokens
                    continue
                if (int(token[2]) - int(mention[2])) <= 200000:  ## Only consider tokens within 200kb of mention
                    context.append(token[1])
            df.append([context, mention[1], mention[3]])
        
        if (i+1) % 1000 == 0:
            print("Finished compiling block", i+1, "of", data_len)
    file.close()
    del data_blocks

    
## Keep only degenerate/ambiguous mentions
## Requiring >=200 occurrences ensures statistical relevance
counts = Counter()
df_len = len(df)
for i,entry in enumerate(df):
    counts.update([entry[2]])
    if (i+1) % 1000 == 0:
        print("Finished counting unique entry", i+1, "of", df_len)
keep_mentions = [url for url,freq in counts.items() if freq >= 200]
df_trimmed = [entry for entry in df if entry[2] in set(keep_mentions)]


## Save the resulting data frame
with open('./data/data.csv','w', encoding = 'utf-8') as file:
    csv.writer(file, delimiter = '\t').writerows(df_trimmed)
    
pickle.dump(df_trimmed, open('./data/data.pkl','wb'))
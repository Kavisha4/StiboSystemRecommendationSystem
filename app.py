from flask import Flask, render_template, request
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import itertools

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("Apriopri/GroceryStoreDataSet.csv", names=['products'], header=None)

# Function for Association Rules
def ar_iterations(data, num_iter=1, support_value=0.1, iterationIndex=None):
  data = list(df["products"].apply(lambda x:x.split(',')))

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_).astype(int)

df

first = pd.DataFrame(df.sum() / df.shape[0], columns = ["Support"]).sort_values("Support", ascending = False)
first

# Elimination by Support Value
first[first.Support >= 0.15]

second = list(itertools.combinations(first.index, 2))
second = [list(i) for i in second]
# Sample of combinations
second[:10]

# Finding support values
value = []
for i in range(0, len(second)):
    temp = df.T.loc[second[i]].sum()
    temp = len(temp[temp == df.T.loc[second[i]].shape[0]]) / df.shape[0]
    value.append(temp)
# Create a data frame
secondIteration = pd.DataFrame(value, columns = ["Support"])
secondIteration["index"] = [tuple(i) for i in second]
secondIteration['length'] = secondIteration['index'].apply(lambda x:len(x))
secondIteration = secondIteration.set_index("index").sort_values("Support", ascending = False)
# Elimination by Support Value
secondIteration = secondIteration[secondIteration.Support > 0.1]
secondIteration



"""# Function for Association Rules"""

def ar_iterations(data, num_iter = 1, support_value = 0.1, iterationIndex = None):

    # Next Iterations
    def ar_calculation(iterationIndex = iterationIndex):
        # Calculation of support value
        value = []
        for i in range(0, len(iterationIndex)):
            result = data.T.loc[iterationIndex[i]].sum()
            result = len(result[result == data.T.loc[iterationIndex[i]].shape[0]]) / data.shape[0]
            value.append(result)
        # Bind results
        result = pd.DataFrame(value, columns = ["Support"])
        result["index"] = [tuple(i) for i in iterationIndex]
        result['length'] = result['index'].apply(lambda x:len(x))
        result = result.set_index("index").sort_values("Support", ascending = False)
        # Elimination by Support Value
        result = result[result.Support > support_value]
        return result

    # First Iteration
    first = pd.DataFrame(df.T.sum(axis = 1) / df.shape[0], columns = ["Support"]).sort_values("Support", ascending = False)
    first = first[first.Support > support_value]
    first["length"] = 1

    if num_iter == 1:
        res = first.copy()

    # Second Iteration
    elif num_iter == 2:

        second = list(itertools.combinations(first.index, 2))
        second = [list(i) for i in second]
        res = ar_calculation(second)

    # All Iterations > 2
    else:
        nth = list(itertools.combinations(set(list(itertools.chain(*iterationIndex))), num_iter))
        nth = [list(i) for i in nth]
        res = ar_calculation(nth)

    return res

iteration1 = ar_iterations(df, num_iter=1, support_value=0.1)
iteration1

iteration2 = ar_iterations(df, num_iter=2, support_value=0.1)
iteration2

iteration3 = ar_iterations(df, num_iter=3, support_value=0.01,
              iterationIndex=iteration2.index)
iteration3

iteration4 = ar_iterations(df, num_iter=4, support_value=0.01,
              iterationIndex=iteration3.index)
iteration4

"""# Association Rules"""

# Apriori
freq_items = apriori(df, min_support = 0.1, use_colnames = True, verbose = 1)
freq_items.sort_values("support", ascending = False)



"""Support value gives us these information:
Head 5

65 percent of 100 purchases are "BREAD"
40 percent of 100 purchases are "COFFEE"
35 percent of 100 purchases are "BISCUIT"
35 percent of 100 purchases are "TEA"
30 percent of 100 purchases are "CORNFLAKES"
"""

# Association Rules & Info
df_ar = association_rules(freq_items, metric = "confidence", min_threshold = 0.5)
df_ar

"""

*   Antecedent support variable tells us probability of antecedent products alone
Consequents support variable tells us probability of consequents products alone
The support value is the value of the two products (Antecedents and Consequents)
Confidence is an indication of how often the rule has been found to be true.
The ratio of the observed support to that expected if X and Y were independent.

"""



df_ar[(df_ar.support > 0.15) & (df_ar.confidence > 0.5)].sort_values("confidence", ascending = False)

# Apriori
freq_items = apriori(df, min_support=0.1, use_colnames=True, verbose=1)

# Association Rules
df_ar = association_rules(freq_items, metric="confidence", min_threshold=0.5)

# Render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Handle form submission
@app.route('/recommend', methods=['POST'])
def recommend():
    product = request.form['product']
    associated_products = get_recommendations(product, df)
    return render_template('index.html', product=product, associated_products=associated_products)

if __name__ == '__main__':
    app.run(debug=True)

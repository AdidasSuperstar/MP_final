
import pandas as pd
import numpy as np

result = pd.read_csv("P:\\Test-f2.csv")

gr = result.groupby(['file_name']).count()
print(gr)

bboxs = np.stack(result['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
print(bboxs)

for i, column in enumerate(['x', 'y', 'w', 'h']):
    result[column] = bboxs[:,i]

result.drop(columns=['bbox'], inplace=True)

result.to_csv("P:\\Test_f2_with_bboxes_pretty.csv")

print(result.columns)
print(result.head())

import numpy as np
import cv2
path="digits.png"
img=cv2.imread(path,0)
cv2.namedWindow("Digit Image", cv2.WINDOW_NORMAL)  # to make the window manually resizeable
cv2.imshow("Digit Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
digit_database.ndim  # digit_database is 4 dimensional
X_train = digit_database[:,:50].reshape(-1,400).astype(np.float32) # Shape = (2500,400)
X_test = digit_database[:,50:100].reshape(-1,400).astype(np.float32) # Shape = (2500,400)
X_train.ndim 
k=np.arange(10)
Y_train=np.repeat(k,3)
Y_train=np.repeat(k,3)[:,np.newaxis]
Y_train

k = np.arange(10)
Y_train = np.repeat(k,250)[:,np.newaxis]
# Since testing data also has data arranged in similar fashion, we are copying the labels
Y_test = Y_train.copy()
knn =  cv2.ml.KNearest_create()  # initilazing
knn.train(X_train,cv2.ml.ROW_SAMPLE,Y_train)  # training model
ret,result,neighbours,dist = knn.findNearest(X_test,k=5)  # predicting results for testing data with K=5

result==Y_test
matches = (result==Y_test)
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)
# Selecting Sample at index 600 and reshaping it to 20x20 to display it
cv2.imshow("Sample 1",X_test[479].reshape(20,20))
cv2.waitKey(0)
cv2.destroyAllWindows()
ret,result,neighbours,dist=knn.findNearest(X_test[479:480],k=5)
print(ret)
print("Digit Recognized as: ",result)
print("Nearest Neighbors: ",neighbours)
print("Distance from Nearest Neighbors: ",dist)
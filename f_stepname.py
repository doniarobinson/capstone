# **Cluster and convert Step Name text values to numerical categories**

# read Step Name into list and save to file 
data_full['Step Name'].fillna('',inplace=True)

# Initialize an empty list to hold the clean reviews
data_cleaned_full = []

# Loop over each row; print index every 100,000 items as confirmation things are still running correctly
print "starting %s" % str(datetime.now())
for i in xrange( 0, num_all_records ):
    if i%100000 == 0:
        print i
    # Call our function for each one, and add the result to the list
    data_cleaned_full.append( data_full.loc[i]['Step Name'] )

print "done %s" % str(datetime.now())

# dump to pickle
joblib.dump(data_cleaned_full, 'feature-files/f_Step_Name.txt')

#################################################################

# reload feature string from pickle file
print "start loading file"
load_step_name = joblib.load('feature-files/f_Step_Name.txt')
print "done loading file"

# Create tf–idf matrix
print "start fitting vectorizer"
step_name_tfidvect = TfidfVectorizer(stop_words = 'english')
X_step_name = step_name_tfidvect.fit_transform(load_step_name)
print "done fitting vectorizer"

# Take a look at the words in the vocabulary
vocab_step_name = step_name_tfidvect.get_feature_names()
#print vocab_step_name

#################################################################

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

print X_step_name.shape[0]

svd_step_name = TruncatedSVD(n_components=2, random_state=15)
print "starting SVD transformation %s" % str(datetime.now())
svd_step_name.fit(X_step_name)

normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd_step_name, normalizer)

# Transform using the TruncatedSVD fit above
data2D_step_name = lsa.transform(X_step_name)
print "done with SVD transformation %s" % str(datetime.now())

#################################################################

k = 200 # Define the number of clusters in which we want to partion the data
print "number of clusters %i" % k
print "starting kmeans %s" % str(datetime.now())

kmeans_step_name = KMeans(n_clusters = k, n_init=3, random_state=15) # Run the algorithm kmeans
kmeans_step_name.fit(data2D_step_name);
centroids_step_name = kmeans_step_name.cluster_centers_ # Get centroid's coordinates for each cluster
labels_step_name = kmeans_step_name.labels_ # Get labels assigned to each data

print "done with kmeans %s" % str(datetime.now())

#################################################################

print load_step_name[1002]
cat_step_name = list(load_step_name)
print cat_step_name[1002]
print (data2D_step_name.shape[0])

print "start saving labels %s" % str(datetime.now())
for i in range(data2D_step_name.shape[0]):
    if i%100000 == 0:
        print i
    cat_step_name[i] = labels_step_name[i]

print "done saving labels %s" % str(datetime.now())
print cat_step_name[1002]

# copy cat_step_name back into full dataframe
data_full['Step Name k200'] = cat_step_name

#################################################################

print "start saving file %s" % str(datetime.now())
filename = 'data/-d' + datetime.now().strftime('%m-%d-%H-%M') + '.txt'
joblib.dump(data_full, filename)
print "done saving file %s" % str(datetime.now())
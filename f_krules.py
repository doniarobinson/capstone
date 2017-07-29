# **Cluster and convert KC(Rules) text values to numerical categories**

# read KC(Rules) into list and save to file 
data_full['KC(Rules)'].fillna('',inplace=True)

# Initialize an empty list to hold the clean reviews
data_cleaned_full = []

# Loop over each row; print index every 100,000 items as confirmation things are still running correctly
print "starting %s" % str(datetime.now())
for i in xrange( 0, num_all_records ):
    if i%100000 == 0:
        print i
    # Call our function for each one, and add the result to the list
    data_cleaned_full.append( data_full.loc[i]['KC(Rules)'] )

print "done %s" % str(datetime.now())

# dump to pickle
joblib.dump(data_cleaned_full, 'feature-files/f_KC(Rules).txt')

#################################################################

# reload feature string from pickle file
print "start loading file"
preload_krules = joblib.load('feature-files/f_KC(Rules).txt')
print "done loading file"

load_krules = map(lambda s: s.replace('~~', '  '), preload_krules)

# Create tf–idf matrix
print "start fitting vectorizer"
tfidvect_krules = TfidfVectorizer(stop_words = 'english')
X_krules_full = tfidvect_krules.fit_transform(load_krules)
print "done fitting vectorizer"

print X_krules_full.shape[0]

# Take a look at the words in the vocabulary
vocab_krules = tfidvect_krules.get_feature_names()
#print vocab_krules

#################################################################

X_krules = X_krules_full.copy()
print X_krules.shape[0]

svd_krules = TruncatedSVD(n_components=5, random_state=15)
print "starting SVD transformation %s" % str(datetime.now())
svd_krules.fit(X_krules) 

# Transform using the TruncatedSVD fit above
data2D_krules = svd_krules.transform(X_krules)
print "done with SVD transformation %s" % str(datetime.now())

k = 200 # Define the number of clusters in which we want to partion the data
print "number of clusters %i" % k
print "starting kmeans %s" % str(datetime.now())

kmeans_krules = KMeans(n_clusters = k, n_init=3, random_state=15) # Run the algorithm kmeans
kmeans_krules.fit(data2D_krules);
centroids_krules = kmeans_krules.cluster_centers_ # Get centroid's coordinates for each cluster
labels_krules = kmeans_krules.labels_ # Get labels assigned to each data

print "done with kmeans %s" % str(datetime.now())

#################################################################

print load_krules[1002]
cat_krules = list(load_krules)
print cat_krules[1002]
print (data2D_krules.shape[0])

print "start saving labels %s" % str(datetime.now())
for i in range(data2D_krules.shape[0]):
    if i%100000 == 0:
        print i
    cat_krules[i] = labels_krules[i]

print "done saving labels %s" % str(datetime.now())

# copy cat_krules back into full dataframe
data_full['KC(Rules) k200'] = cat_krules

print "start saving file %s" % str(datetime.now())
filename = 'data/-d' + datetime.now().strftime('%m-%d-%H-%M') + '.txt'
joblib.dump(data_full, filename)
print "done saving file %s" % str(datetime.now())
# **Cluster and convert KC(SubSkills) text values to numerical categories**

# read KC(SubSkills) into list and save to file 
data_full['KC(SubSkills)'].fillna('',inplace=True)

# Initialize an empty list
data_cleaned_full = []

# Loop over each row; print index every 100,000 items as confirmation things are still running correctly
print "starting %s" % str(datetime.now())
for i in xrange( 0, num_all_records ):
    if i%100000 == 0:
        print i
    # Call our function for each one, and add the result to the list
    data_cleaned_full.append( data_full.loc[i]['KC(SubSkills)'] )

print "done %s" % str(datetime.now())

# dump to pickle
joblib.dump(data_cleaned_full, 'feature-files/f_KC(SubSkills).txt')

#################################################################

# reload feature string from pickle file
print "start loading file"
preload_ksubskills = joblib.load('feature-files/f_KC(SubSkills).txt')
print "done loading file"

load_ksubskills = map(lambda s: s.replace('~~', '  '), preload_ksubskills)

# Create tf–idf matrix
print "start fitting vectorizer"
tfidvect_ksubskills = TfidfVectorizer(stop_words = 'english')
X_ksubskills_full = tfidvect_ksubskills.fit_transform(load_ksubskills)
print "done fitting vectorizer"

print X_ksubskills_full.shape[0]

# Take a look at the words in the vocabulary
vocab_ksubskills = tfidvect_ksubskills.get_feature_names()
#print vocab_ksubskills

#################################################################

X_ksubskills = X_ksubskills_full.copy()
print X_ksubskills.shape[0]

svd_ksubskills = TruncatedSVD(n_components=5, random_state=15)
print "starting SVD transformation %s" % str(datetime.now())
svd_ksubskills.fit(X_ksubskills) 

# Transform using the TruncatedSVD fit above
data2D_ksubskills = svd_ksubskills.transform(X_ksubskills)
print "done with SVD transformation %s" % str(datetime.now())

k = 200 # Define the number of clusters in which we want to partion the data
print "number of clusters %i" % k
print "starting kmeans %s" % str(datetime.now())

kmeans_ksubskills = KMeans(n_clusters = k, n_init=3, random_state=15) # Run the algorithm kmeans
kmeans_ksubskills.fit(data2D_ksubskills);
centroids_ksubskills = kmeans_ksubskills.cluster_centers_ # Get centroid's coordinates for each cluster
labels_ksubskills = kmeans_ksubskills.labels_ # Get labels assigned to each data

print "done with kmeans %s" % str(datetime.now())

#################################################################

print load_ksubskills[1002]
cat_ksubskills = list(load_ksubskills)
print cat_ksubskills[1002]
print (data2D_ksubskills.shape[0])

print "start saving labels %s" % str(datetime.now())
for i in range(data2D_ksubskills.shape[0]):
    if i%100000 == 0:
        print i
    cat_ksubskills[i] = labels_ksubskills[i]

print "done saving labels %s" % str(datetime.now())

# copy cat_ksubskills back into full dataframe
data_full['KC(SubSkills) k200'] = cat_ksubskills

print "start saving file %s" % str(datetime.now())
filename = 'data/-d' + datetime.now().strftime('%m-%d-%H-%M') + '.txt'
joblib.dump(data_full, filename)
print "done saving file %s" % str(datetime.now())
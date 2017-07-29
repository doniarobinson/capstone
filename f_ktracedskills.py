# **Cluster and convert KC(KTracedSkills) text values to numerical categories**

# read KC(KTracedSkills) into list and save to file 
data_full['KC(KTracedSkills)'].fillna('',inplace=True)

# Initialize an empty list to hold the clean reviews
data_cleaned_full = []

# Loop over each row; print index every 100,000 items as confirmation things are still running correctly
print "starting %s" % str(datetime.now())
for i in xrange( 0, num_all_records ):
    if i%100000 == 0:
        print i
    # Call our function for each one, and add the result to the list
    data_cleaned_full.append( data_full.loc[i]['KC(KTracedSkills)'] )

print "done %s" % str(datetime.now())

# dump to pickle
joblib.dump(data_cleaned_full, 'feature-files/f_KC(KTracedSkills).txt')

#################################################################

# reload feature string from pickle file
print "start loading file"
preload_ktracedskills = joblib.load('feature-files/f_KC(KTracedSkills).txt')
print "done loading file"

load_ktracedskills = map(lambda s: s.replace('~~', '  '), preload_ktracedskills)

# Create tf–idf matrix
print "start fitting vectorizer"
ktracedskills_tfidvect = TfidfVectorizer(stop_words = 'english')
X_ktracedskills = ktracedskills_tfidvect.fit_transform(load_ktracedskills)
print "done fitting vectorizer"

# Take a look at the words in the vocabulary
vocab_ktracedskills = ktracedskills_tfidvect.get_feature_names()
print vocab_ktracedskills

#################################################################

print X_ktracedskills.shape[0]

svd_ktracedskills = TruncatedSVD(n_components=5, random_state=15)
print "starting SVD transformation %s" % str(datetime.now())
svd_ktracedskills.fit(X_ktracedskills) 

# Transform using the TruncatedSVD fit above
data2D_ktracedskills = svd_ktracedskills.transform(X_ktracedskills)
print "done with SVD transformation %s" % str(datetime.now())

#################################################################

k = 200 # Define the number of clusters in which we want to partion the data
print "number of clusters %i" % k
print "starting kmeans %s" % str(datetime.now())

kmeans_ktracedskills = KMeans(n_clusters = k, n_init=3, random_state=15) # Run the algorithm kmeans
kmeans_ktracedskills.fit(data2D_ktracedskills);
centroids_ktracedskills = kmeans_ktracedskills.cluster_centers_ # Get centroid's coordinates for each cluster
labels_ktracedskills = kmeans_ktracedskills.labels_ # Get labels assigned to each data

print "done with kmeans %s" % str(datetime.now())

#################################################################

print load_ktracedskills[1002]
cat_ktracedskills = list(load_ktracedskills)
print cat_ktracedskills[1002]
print (data2D_ktracedskills.shape[0])

print "start saving labels %s" % str(datetime.now())
for i in range(data2D_ktracedskills.shape[0]):
    if i%100000 == 0:
        print i
    cat_ktracedskills[i] = labels_ktracedskills[i]

print "done saving labels %s" % str(datetime.now())
print cat_ktracedskills[1002]

# copy cat_ktracedskills back into full dataframe
data_full['KC(KTracedSkills) k200'] = cat_ktracedskills

#################################################################

print "start saving file %s" % str(datetime.now())
filename = 'data/-d' + datetime.now().strftime('%m-%d-%H-%M') + '.txt'
joblib.dump(data_full, filename)
print "done saving file %s" % str(datetime.now())
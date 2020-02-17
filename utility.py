import pickle

#Saving object
def save_obj(obj,name):
    print ("Writing",name,"File to disk")
    with open('obj1/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
    print("Writing done")

#Load saved object file
def load_obj(name):
    print("Loading obj :",name)
    with open('obj1/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj_text(obj,name):
    print ("Writing", name, "File to disk")
    with open('obj1/' + name + '.txt', 'w') as f:
        pickle.dump(obj, f)
        # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("Writing done")
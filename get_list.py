import os

dataset = "office-home"
dataset_path = "."
if dataset == "office-31":
    domains = ["amazon", "dslr", "webcam"]
elif dataset == "office-caltech":
    domains = ["amazon", "dslr", "webcam", "caltech"]
elif dataset == "office-home":
    domains = ["Art", "Clipart", "Product", "Real_World"]
elif dataset == "domainnet":
    domains = ["real", "infograph", "painting", "sketch", "clipart", "quickdraw"]
else:
    print("No such dataset exists!")

for domain in domains:
    log = open(dataset_path + "/" + dataset + "/" + domain + "_list.txt", "w")
    directory = os.path.join(dataset_path, dataset, domain)
    print(directory)
    classes = [x[1] for x in os.walk(directory)]
    print(classes)
    classes = classes[0]
    classes.sort()
    for idx, f in enumerate(classes):
        files = os.listdir(dataset_path + "/" + dataset + "/" + domain + "/" + f)
        for file in files:
            s = (
                os.path.join(dataset_path + "/" + dataset, domain, f, file)
                + " "
                + str(idx)
                + "\n"
            )
            log.write(s)
    log.close()

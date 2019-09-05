import os, sys

def glob_by_ext(dir_path, exts):
    results = []
    if os.path.isdir(dir_path):
        for name in os.listdir(dir_path):
            if name.startswith('.'): continue
            base, ext = os.path.splitext(name)
            if ext.lower() in exts:
                results.append(name)
    return results

if __name__ == "__main__":
    origin_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190317_static_clothing/Cloth_Delivery"
    label_dir = "/Volumes/ClothesData/20190401_Data_Clothing/20190806_labeled_clothing/Static"

    to_remove = []

    for name in os.listdir(origin_dir):
        origin_path = os.path.join(origin_dir, name)
        label_path = os.path.join(label_dir, name)
        if os.path.isdir(origin_path):
            if not os.path.exists(label_path):
                print("Error: %s is missing" % name)
                continue

            origin_models = glob_by_ext(origin_path, exts=['.obj', '.ply', '.off'])
            label_models = glob_by_ext(label_path, exts=['.obj', '.ply', '.off'])

            if len(origin_models) == 0:
                print("Error: Origin directory has no models: %s" % name)
                continue

            if len(origin_models) != 1:
                print("Error: Origin directory has multiple models: %s" % name)
                print("All origin models listed below:")
                print('\n'.join(origin_models))
                continue

            if origin_models[0] not in label_models:
                print("Error: Origin model does not exist in labeld directory: %s" % name)

            redundancy = [ os.path.join(label_path, n) for n in label_models if n not in origin_models ]
            to_remove.extend(redundancy)
            if len(redundancy) != 0:
                print("Error: Redundant models detected: %s" % name)

            origin_mtls = glob_by_ext(origin_path, exts=['.mtl'])
            label_mtls = glob_by_ext(label_path, exts=['.mtl'])

            if len(origin_mtls) != 1:
                print("Error: Origin directory has multiple materials: %s" % name)
                continue

            if origin_mtls[0] not in label_mtls:
                print("Error: Origin materials does not exist in label directory: %s" % name)

            redundancy = [ os.path.join(label_path, n) for n in label_mtls if n not in origin_mtls ]
            to_remove.extend(redundancy)
            if len(redundancy) != 0:
                print("Error: Redundant materials detected: %s" % name)

            components = ['top', 'bottom', 'shoes']
            for component in components:
                component_path = os.path.join(label_path, component)
                component_models = glob_by_ext(component_path, exts=['.obj', '.ply', '.off'])
                if len(component_models) >= 2:
                    print("Error: Redundant component models detected: %s/%s" % (name, component))
                    print("All component models listed below:")
                    print('\n'.join(component_models))

                component_mtls = glob_by_ext(component_path, exts=['.mtl'])
                if len(component_mtls) >= 2:
                    print("Error: Redundant component materials detected: %s/%s" % (name, component))
                    print("All component materials listed below:")
                    print('\n'.join(component_mtls))

    if len(to_remove) != 0:
        print("Some redundancy detected:")
        print('\n'.join(to_remove))
        while True:
            confirm = input("Remove them right now? (y/n)").lower()
            if confirm == 'yes' or confirm == 'y':
                for path in to_remove:
                    print("Removing %s" % path)
                    os.remove(path)
                break
            elif confirm == 'no' or confirm == 'n':
                break
            else:
                print("Please input y/n instead!")
                continue


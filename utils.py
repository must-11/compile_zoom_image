ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])

def allwed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

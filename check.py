import mimetypes
import os

import magic

filename = r".dvc\cache\3e\a8e5d4f415ddba6a294bee61e55db6"

# Detect file extension
file_extension = os.path.splitext(filename)[1]
print(f"File extension: {file_extension}")

# Detect MIME type using mimetypes module
mime_type, _ = mimetypes.guess_type(filename)
print(f"MIME type (mimetypes): {mime_type}")

# Detect MIME type using python-magic
mime_type_magic = magic.from_file(filename, mime=True)
print(f"MIME type (python-magic): {mime_type_magic}")

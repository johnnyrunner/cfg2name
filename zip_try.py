import shutil

zip_name = 'zip_try_now'
directory_name = 'path\to\directory'

# Create 'path\to\zip_file.zip'
shutil.make_archive(zip_name, 'zip', directory_name)

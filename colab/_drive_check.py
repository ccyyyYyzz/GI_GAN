import os
p='/content/drive/MyDrive/ns_mc_gan_gi/vqgan_rate_repo_bundle.zip'
print('DRIVE_BUNDLE_EXISTS', os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else 0)

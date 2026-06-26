import sys
print(sys.version)
for mod in ['torch','torchvision','lpips','skimage','yaml']:
    try:
        m=__import__(mod)
        print(mod, getattr(m, '__version__', 'ok'))
        if mod=='torch':
            print('cuda', m.cuda.is_available(), m.cuda.get_device_name(0) if m.cuda.is_available() else None)
    except Exception as e:
        print(mod+'_error', repr(e))
require 'image'

all_fnames = paths.dir('examples/')
demo_fnames = {}
fake_fnames = {}
for i=1,#all_fnames do 
  if string.match(all_fnames[i], "demo") then
    table.insert(demo_fnames, all_fnames[i])
  end
  if string.match(all_fnames[i], "fake") then
    table.insert(fake_fnames, all_fnames[i])
  end
end

for i=1,#demo_fnames do
    src=image.load(string.format('examples/%s',string.sub(fake_fnames[i], 6, #fake_fnames[i])))
    tgt=image.load(string.format('examples/%s',demo_fnames[i]))
    mask=image.load('mask.png')
    mask=mask[{{1,3},{},{}}]
    res=torch.cmul(src,mask)+torch.cmul(tgt,1-mask)
    image.save(string.format('examples/result_%s.png',i),res)
end

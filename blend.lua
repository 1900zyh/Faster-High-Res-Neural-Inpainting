require 'image'

all_fnames = paths.dir('examples/')
demo_fnames = {}
fake_fnames = {}
for i=1,#all_fnames do 
  if string.match(all_fnames[i], "demo") then
    table.insert(demo_fnames, all_fnames[i])
  else
    if string.match(all_fnames[i], "fake") then
      table.insert(fake_fnames, all_fnames[i])
    end
  end
end

table.sort(fake_fnames)
table.sort(demo_fnames)

for i=1,#demo_fnames do
    src=image.load(string.format('examples/%s',string.sub(fake_fnames[i], 6, #fake_fnames[i])))
    tgt=image.load(string.format('examples/%s',demo_fnames[i]))
    mask=image.load('mask.png')
    mask=mask[{{1,3},{},{}}]
    res=torch.cmul(src,mask)+torch.cmul(tgt,1-mask)
    image.save(string.format('examples/result_%s.png',string.sub(fake_fnames[i], 6, #fake_fnames[i]-4)),res)
end

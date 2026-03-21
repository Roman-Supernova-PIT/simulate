# simulate
Simulation Helper

> [!CAUTION]
> The code in this repository is under heavy development and not currently intended for widespread use.
>

## Example useage:

Create a new environment using the requirements file:
  `conda env create -f requirements.yaml`

## Look at the config file
`sim2.config` -> edit for things you want

## Run basic sim included with main
This will make a few level 2 images
`python simulate.py --plot --input_config_file sim2.config --level 2`

You should see as output these files:
```

```

### With the default star, gaia, and galaxy catalogs
You should get images that look something like the following:
<img width="800" height="800" alt="r0003201001001001007_0003_wfi01_f106_cal" src="https://github.com/user-attachments/assets/d201d974-16dd-4739-b237-aab581af817b" />

<img width="800" height="800" alt="r0003201001001001006_0003_wfi01_f106_cal" src="https://github.com/user-attachments/assets/e29571b1-f965-4bae-b0cf-bae99d0a7d87" />


You can create a PNG from any of the images using the imagelib.py:
```
import imagelib
imagelib.mkfigure('myfile.asdf',plotname='myfile.png')
```

> [!NOTE]
> F129 is giving very low flux in the objects
>
> I haven't gotten L1 images to run through romancal yet at the versions installed in the requirements file

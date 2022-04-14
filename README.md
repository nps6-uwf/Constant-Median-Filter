# Constant-Median-Filter
A cython implementation of the constant time median filter.


<img src='https://github.com/nps6-uwf/Constant-Median-Filter/blob/main/results/Figure_2.png?raw=true'>
<p><b>Figure 1.</b>  A pit viper I saw in Thailand.  The <i>hue</i> (first), <i>saturation</i> (second), and <i>value</i> (third) channels are isolated, blurred, and then merged.</p>

<img src='https://github.com/nps6-uwf/Constant-Median-Filter/blob/main/results/Figure_4.png?raw=true'>
  <p><b>Figure 2.</b>  A zebra.  The <i>hue</i> (first), <i>saturation</i> (second), and <i>value</i> (third) channels are isolated, blurred, and then merged.</p>

<img src='https://github.com/nps6-uwf/Constant-Median-Filter/blob/main/results/Figure_5.png?raw=true'>
<p><b>Figure 3.</b>  A trio of tadpoles.  The <i>red</i> (first), <i>green</i> (second), and <i>blue</i> (third) channels are isolated, blurred, and then merged.</p>

## Source Images
<table>
  <tr>
    <td>
      <img src='https://github.com/nps6-uwf/Constant-Median-Filter/blob/main/results/zebra.png?raw=true'>
    </td>
    <td>
    </td>
    <tr>
</table>

## Usage
1. Compile the cython 
<code>python setup.py build_ext --inplace</code>

2. Import into program
<code>from ctmf import ctmf</code>

Or you can use the pure python version, but its slow and only useful for educational purposes.  

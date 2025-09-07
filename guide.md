NVIDIA 确认：cuda-toolkit 是元包（meta-package），设计上会拉取最新兼容依赖

官方建议：需要精确版本时改用 cudatoolkit

conda search -c nvidia cudatoolkit
Loading channels: done
# Name                       Version           Build  Channel             
cudatoolkit                      9.0      h13b8566_0  anaconda/pkgs/main  
cudatoolkit                      9.2               0  nvidia              
cudatoolkit                      9.2               0  anaconda/pkgs/main  
cudatoolkit                  9.2.148     h33e3169_12  conda-forge         
cudatoolkit                  9.2.148     h33e3169_13  conda-forge         
cudatoolkit                  9.2.148     h60dc4a4_10  conda-forge         
cudatoolkit                  9.2.148     h60dc4a4_11  conda-forge         
cudatoolkit                  9.2.148     h80a95b2_10  conda-forge         
cudatoolkit                  9.2.148      h80a95b2_6  conda-forge         
cudatoolkit                  9.2.148      h80a95b2_7  conda-forge         
cudatoolkit                  9.2.148      h80a95b2_8  conda-forge         
cudatoolkit                  9.2.148      h80a95b2_9  conda-forge         
cudatoolkit                 10.0.130               0  nvidia              
cudatoolkit                 10.0.130               0  anaconda/pkgs/main  
cudatoolkit                 10.0.130     h8c5a6a4_10  conda-forge         
cudatoolkit                 10.0.130     h8c5a6a4_11  conda-forge         
cudatoolkit                 10.0.130     h9ed11e1_12  conda-forge         
cudatoolkit                 10.0.130     h9ed11e1_13  conda-forge         
cudatoolkit                 10.0.130     hf841e97_10  conda-forge         
cudatoolkit                 10.0.130      hf841e97_6  conda-forge         
cudatoolkit                 10.0.130      hf841e97_7  conda-forge         
cudatoolkit                 10.0.130      hf841e97_8  conda-forge         
cudatoolkit                 10.0.130      hf841e97_9  conda-forge         
cudatoolkit                 10.1.168               0  anaconda/pkgs/main  
cudatoolkit                 10.1.243     h036e899_10  conda-forge         
cudatoolkit                 10.1.243      h036e899_6  conda-forge         
cudatoolkit                 10.1.243      h036e899_7  conda-forge         
cudatoolkit                 10.1.243      h036e899_8  nvidia              
cudatoolkit                 10.1.243      h036e899_8  conda-forge         
cudatoolkit                 10.1.243      h036e899_9  conda-forge         
cudatoolkit                 10.1.243      h6bb024c_0  nvidia              
cudatoolkit                 10.1.243      h6bb024c_0  anaconda/pkgs/main  
cudatoolkit                 10.1.243     h6d9799a_12  conda-forge         
cudatoolkit                 10.1.243     h6d9799a_13  conda-forge         
cudatoolkit                 10.1.243     h8cb64d8_10  conda-forge         
cudatoolkit                 10.1.243     h8cb64d8_11  conda-forge         
cudatoolkit                  10.2.89      h6bb024c_0  nvidia              
cudatoolkit                  10.2.89     h713d32c_10  conda-forge         
cudatoolkit                  10.2.89     h713d32c_11  conda-forge         
cudatoolkit                  10.2.89     h8f6ccaa_10  conda-forge         
cudatoolkit                  10.2.89      h8f6ccaa_6  conda-forge         
cudatoolkit                  10.2.89      h8f6ccaa_7  conda-forge         
cudatoolkit                  10.2.89      h8f6ccaa_8  nvidia              
cudatoolkit                  10.2.89      h8f6ccaa_8  conda-forge         
cudatoolkit                  10.2.89      h8f6ccaa_9  conda-forge         
cudatoolkit                  10.2.89     hdec6ad0_12  conda-forge         
cudatoolkit                  10.2.89     hdec6ad0_13  conda-forge         
cudatoolkit                  10.2.89      hfd86e86_0  anaconda/pkgs/main  
cudatoolkit                  10.2.89      hfd86e86_1  anaconda/pkgs/main  
cudatoolkit                   11.0.3     h15472ef_10  conda-forge         
cudatoolkit                   11.0.3      h15472ef_6  conda-forge         
cudatoolkit                   11.0.3      h15472ef_7  conda-forge         
cudatoolkit                   11.0.3      h15472ef_8  nvidia              
cudatoolkit                   11.0.3      h15472ef_8  conda-forge         
cudatoolkit                   11.0.3      h15472ef_9  conda-forge         
cudatoolkit                   11.0.3     h7761cd4_12  conda-forge         
cudatoolkit                   11.0.3     h7761cd4_13  conda-forge         
cudatoolkit                   11.0.3     h88f8997_10  conda-forge         
cudatoolkit                   11.0.3     h88f8997_11  conda-forge         
cudatoolkit                 11.0.221      h6bb024c_0  nvidia              
cudatoolkit                 11.0.221      h6bb024c_0  anaconda/pkgs/main  
cudatoolkit                   11.1.1     h6406543_10  conda-forge         
cudatoolkit                   11.1.1      h6406543_6  conda-forge         
cudatoolkit                   11.1.1      h6406543_7  conda-forge         
cudatoolkit                   11.1.1      h6406543_8  nvidia              
cudatoolkit                   11.1.1      h6406543_8  conda-forge         
cudatoolkit                   11.1.1      h6406543_9  conda-forge         
cudatoolkit                   11.1.1     ha002fc5_10  conda-forge         
cudatoolkit                   11.1.1     ha002fc5_11  conda-forge         
cudatoolkit                   11.1.1     hb139c0e_12  conda-forge         
cudatoolkit                   11.1.1     hb139c0e_13  conda-forge         
cudatoolkit                  11.1.74      h6bb024c_0  nvidia              
cudatoolkit                   11.2.0      h73cb219_7  conda-forge         
cudatoolkit                   11.2.0      h73cb219_8  nvidia              
cudatoolkit                   11.2.0      h73cb219_8  conda-forge         
cudatoolkit                   11.2.0      h73cb219_9  conda-forge         
cudatoolkit                   11.2.1      h8204236_8  nvidia              
cudatoolkit                   11.2.1      h8204236_8  conda-forge         
cudatoolkit                   11.2.1      h8204236_9  conda-forge         
cudatoolkit                   11.2.2     hbe64b41_10  conda-forge         
cudatoolkit                   11.2.2     hbe64b41_11  conda-forge         
cudatoolkit                   11.2.2     hc23eb0c_12  conda-forge         
cudatoolkit                   11.2.2     hc23eb0c_13  conda-forge         
cudatoolkit                   11.2.2     he111cf0_10  conda-forge         
cudatoolkit                   11.2.2      he111cf0_8  nvidia              
cudatoolkit                   11.2.2      he111cf0_8  conda-forge         
cudatoolkit                   11.2.2      he111cf0_9  conda-forge         
cudatoolkit                  11.2.72      h2bc3f7f_0  nvidia              
cudatoolkit                   11.3.1      h2bc3f7f_2  anaconda/pkgs/main  
cudatoolkit                   11.3.1     h9edb442_10  conda-forge         
cudatoolkit                   11.3.1     h9edb442_11  conda-forge         
cudatoolkit                   11.3.1     ha36c431_10  conda-forge         
cudatoolkit                   11.3.1      ha36c431_9  nvidia              
cudatoolkit                   11.3.1      ha36c431_9  conda-forge         
cudatoolkit                   11.3.1     hb98b00a_12  conda-forge         
cudatoolkit                   11.3.1     hb98b00a_13  conda-forge         
cudatoolkit                   11.4.0      hebf7ef1_9  nvidia              
cudatoolkit                   11.4.1      h8ab8bb3_9  nvidia              
cudatoolkit                   11.4.2     h00f7ccd_10  conda-forge         
cudatoolkit                   11.4.2      h00f7ccd_9  conda-forge         
cudatoolkit                   11.4.2     h7a5bcfd_10  conda-forge         
cudatoolkit                   11.4.2     h7a5bcfd_11  conda-forge         
cudatoolkit                   11.4.3     h39f8164_12  conda-forge         
cudatoolkit                   11.4.3     h39f8164_13  conda-forge         
cudatoolkit                   11.5.0      h36ae40a_9  nvidia              
cudatoolkit                   11.5.0      h36ae40a_9  conda-forge         
cudatoolkit                   11.5.1     h59c8dcf_10  conda-forge         
cudatoolkit                   11.5.1     h59c8dcf_11  conda-forge         
cudatoolkit                   11.5.1     hcf5317a_10  conda-forge         
cudatoolkit                   11.5.1      hcf5317a_9  nvidia              
cudatoolkit                   11.5.1      hcf5317a_9  conda-forge         
cudatoolkit                   11.5.2     hbdc67f6_12  conda-forge         
cudatoolkit                   11.5.2     hbdc67f6_13  conda-forge         
cudatoolkit                   11.6.0     habf752d_10  conda-forge         
cudatoolkit                   11.6.0      habf752d_9  nvidia              
cudatoolkit                   11.6.0      habf752d_9  conda-forge         
cudatoolkit                   11.6.0     hecad31d_10  conda-forge         
cudatoolkit                   11.6.0     hecad31d_11  conda-forge         
cudatoolkit                   11.6.1     h775ab47_12  conda-forge         
cudatoolkit                   11.6.1     h775ab47_13  conda-forge         
cudatoolkit                   11.6.2     hfc3e2af_12  conda-forge         
cudatoolkit                   11.6.2     hfc3e2af_13  conda-forge         
cudatoolkit                   11.7.0     hd8887f6_10  nvidia              
cudatoolkit                   11.7.0     hd8887f6_10  conda-forge         
cudatoolkit                   11.7.0     hd8887f6_11  conda-forge         
cudatoolkit                   11.7.1     h4bc3d14_12  conda-forge         
cudatoolkit                   11.7.1     h4bc3d14_13  conda-forge         
cudatoolkit                   11.8.0     h37601d7_10  conda-forge         
cudatoolkit                   11.8.0     h37601d7_11  conda-forge         
cudatoolkit                   11.8.0     h4ba93d1_12  conda-forge         
cudatoolkit                   11.8.0     h4ba93d1_13  conda-forge         
cudatoolkit                   11.8.0      h6a678d5_0  anaconda/pkgs/main 

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

conda install -c nvidia cuda-toolkit=12.8 相当于 cudatoolkit-dev + cudatoolkit
conda install -c nvidia cudatoolkit-dev
conda install -c nvidia cudnn=8.9.6

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

conda install -c conda-forge gxx_linux-64=10.3.0 cxx-compiler=1.4.0 -y


conda install -c nvidia cuda-toolkit=11.3
conda install -c nvidia cudatoolkit-dev=11.3
conda install -c nvidia cudnn=8.2.1

conda install -c conda-forge --override-channels \
    eigen=3.4.0 \
    boost \
    boost-cpp



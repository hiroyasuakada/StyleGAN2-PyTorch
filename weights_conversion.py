import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightsConverter():
    def __init__(self):

        #   'style_mixing_rate'        : ['uns', 'lod'                                          ],
        self.name_trans_dict = {
            'synthesis_network.init_block.const_input'                      : ['any', 'G_synthesis/4x4/Const/const'                  ],

            'mapping_network.blocks.1.fc.weight'                            : ['fc_', 'G_mapping/Dense0/weight'                      ],
            'mapping_network.blocks.1.bias.bias'                            : ['any', 'G_mapping/Dense0/bias'                        ],
            'mapping_network.blocks.2.fc.weight'                            : ['fc_', 'G_mapping/Dense1/weight'                      ],
            'mapping_network.blocks.2.bias.bias'                            : ['any', 'G_mapping/Dense1/bias'                        ],
            'mapping_network.blocks.3.fc.weight'                            : ['fc_', 'G_mapping/Dense2/weight'                      ],
            'mapping_network.blocks.3.bias.bias'                            : ['any', 'G_mapping/Dense2/bias'                        ],
            'mapping_network.blocks.4.fc.weight'                            : ['fc_', 'G_mapping/Dense3/weight'                      ],
            'mapping_network.blocks.4.bias.bias'                            : ['any', 'G_mapping/Dense3/bias'                        ],
            'mapping_network.blocks.5.fc.weight'                            : ['fc_', 'G_mapping/Dense4/weight'                      ],
            'mapping_network.blocks.5.bias.bias'                            : ['any', 'G_mapping/Dense4/bias'                        ],
            'mapping_network.blocks.6.fc.weight'                            : ['fc_', 'G_mapping/Dense5/weight'                      ],
            'mapping_network.blocks.6.bias.bias'                            : ['any', 'G_mapping/Dense5/bias'                        ],
            'mapping_network.blocks.7.fc.weight'                            : ['fc_', 'G_mapping/Dense6/weight'                      ],
            'mapping_network.blocks.7.bias.bias'                            : ['any', 'G_mapping/Dense6/bias'                        ],
            'mapping_network.blocks.8.fc.weight'                            : ['fc_', 'G_mapping/Dense7/weight'                      ],
            'mapping_network.blocks.8.bias.bias'                            : ['any', 'G_mapping/Dense7/bias'                        ],
            'mapping_network.blocks.9.avg_style'                            : ['any', 'dlatent_avg'                                  ],
 
            'synthesis_network.init_block.conv.conv.weight'                 : ['con', 'G_synthesis/4x4/Conv/weight'                  ],
            'synthesis_network.init_block.conv.conv.fc.weight'              : ['fc_', 'G_synthesis/4x4/Conv/mod_weight'              ],
            'synthesis_network.init_block.conv.conv.bias.bias'              : ['any', 'G_synthesis/4x4/Conv/mod_bias'                ],
            'synthesis_network.init_block.conv.noise.noise_scaler'          : ['uns', 'G_synthesis/4x4/Conv/noise_strength'          ],
            'synthesis_network.init_block.conv.noise.const_noise'           : ['any', 'G_synthesis/noise0'                           ],
            'synthesis_network.init_block.conv.bias.bias'                   : ['any', 'G_synthesis/4x4/Conv/bias'                    ],

            'synthesis_network.blocks.0.conv0_up.conv.weight'               : ['mTc', 'G_synthesis/8x8/Conv0_up/weight'              ],
            'synthesis_network.blocks.0.conv0_up.conv.fc.weight'            : ['fc_', 'G_synthesis/8x8/Conv0_up/mod_weight'          ],
            'synthesis_network.blocks.0.conv0_up.conv.bias.bias'            : ['any', 'G_synthesis/8x8/Conv0_up/mod_bias'            ],
            'synthesis_network.blocks.0.conv0_up.noise.noise_scaler'        : ['uns', 'G_synthesis/8x8/Conv0_up/noise_strength'      ],
            'synthesis_network.blocks.0.conv0_up.noise.const_noise'         : ['any', 'G_synthesis/noise1'                           ],
            'synthesis_network.blocks.0.conv0_up.bias.bias'                 : ['any', 'G_synthesis/8x8/Conv0_up/bias'                ],
            'synthesis_network.blocks.0.conv1.conv.weight'                  : ['con', 'G_synthesis/8x8/Conv1/weight'                 ],
            'synthesis_network.blocks.0.conv1.conv.fc.weight'               : ['fc_', 'G_synthesis/8x8/Conv1/mod_weight'             ],
            'synthesis_network.blocks.0.conv1.conv.bias.bias'               : ['any', 'G_synthesis/8x8/Conv1/mod_bias'               ],
            'synthesis_network.blocks.0.conv1.noise.noise_scaler'           : ['uns', 'G_synthesis/8x8/Conv1/noise_strength'         ],
            'synthesis_network.blocks.0.conv1.noise.const_noise'            : ['any', 'G_synthesis/noise2'                           ],
            'synthesis_network.blocks.0.conv1.bias.bias'                    : ['any', 'G_synthesis/8x8/Conv1/bias'                   ],

            'synthesis_network.blocks.1.conv0_up.conv.weight'               : ['mTc', 'G_synthesis/16x16/Conv0_up/weight'            ],
            'synthesis_network.blocks.1.conv0_up.conv.fc.weight'            : ['fc_', 'G_synthesis/16x16/Conv0_up/mod_weight'        ],
            'synthesis_network.blocks.1.conv0_up.conv.bias.bias'            : ['any', 'G_synthesis/16x16/Conv0_up/mod_bias'          ],
            'synthesis_network.blocks.1.conv0_up.noise.noise_scaler'        : ['uns', 'G_synthesis/16x16/Conv0_up/noise_strength'    ],
            'synthesis_network.blocks.1.conv0_up.noise.const_noise'         : ['any', 'G_synthesis/noise3'                           ],
            'synthesis_network.blocks.1.conv0_up.bias.bias'                 : ['any', 'G_synthesis/16x16/Conv0_up/bias'              ],
            'synthesis_network.blocks.1.conv1.conv.weight'                  : ['con', 'G_synthesis/16x16/Conv1/weight'               ],
            'synthesis_network.blocks.1.conv1.conv.fc.weight'               : ['fc_', 'G_synthesis/16x16/Conv1/mod_weight'           ],
            'synthesis_network.blocks.1.conv1.conv.bias.bias'               : ['any', 'G_synthesis/16x16/Conv1/mod_bias'             ],
            'synthesis_network.blocks.1.conv1.noise.noise_scaler'           : ['uns', 'G_synthesis/16x16/Conv1/noise_strength'       ],
            'synthesis_network.blocks.1.conv1.noise.const_noise'            : ['any', 'G_synthesis/noise4'                           ],
            'synthesis_network.blocks.1.conv1.bias.bias'                    : ['any', 'G_synthesis/16x16/Conv1/bias'                 ],

            'synthesis_network.blocks.2.conv0_up.conv.weight'               : ['mTc', 'G_synthesis/32x32/Conv0_up/weight'            ],
            'synthesis_network.blocks.2.conv0_up.conv.fc.weight'            : ['fc_', 'G_synthesis/32x32/Conv0_up/mod_weight'        ],
            'synthesis_network.blocks.2.conv0_up.conv.bias.bias'            : ['any', 'G_synthesis/32x32/Conv0_up/mod_bias'          ],
            'synthesis_network.blocks.2.conv0_up.noise.noise_scaler'        : ['uns', 'G_synthesis/32x32/Conv0_up/noise_strength'    ],
            'synthesis_network.blocks.2.conv0_up.noise.const_noise'         : ['any', 'G_synthesis/noise5'                           ],
            'synthesis_network.blocks.2.conv0_up.bias.bias'                 : ['any', 'G_synthesis/32x32/Conv0_up/bias'              ],
            'synthesis_network.blocks.2.conv1.conv.weight'                  : ['con', 'G_synthesis/32x32/Conv1/weight'               ],
            'synthesis_network.blocks.2.conv1.conv.fc.weight'               : ['fc_', 'G_synthesis/32x32/Conv1/mod_weight'           ],
            'synthesis_network.blocks.2.conv1.conv.bias.bias'               : ['any', 'G_synthesis/32x32/Conv1/mod_bias'             ],
            'synthesis_network.blocks.2.conv1.noise.noise_scaler'           : ['uns', 'G_synthesis/32x32/Conv1/noise_strength'       ],
            'synthesis_network.blocks.2.conv1.noise.const_noise'            : ['any', 'G_synthesis/noise6'                           ],
            'synthesis_network.blocks.2.conv1.bias.bias'                    : ['any', 'G_synthesis/32x32/Conv1/bias'                 ],

            'synthesis_network.blocks.3.conv0_up.conv.weight'               : ['mTc', 'G_synthesis/64x64/Conv0_up/weight'            ],
            'synthesis_network.blocks.3.conv0_up.conv.fc.weight'            : ['fc_', 'G_synthesis/64x64/Conv0_up/mod_weight'        ],
            'synthesis_network.blocks.3.conv0_up.conv.bias.bias'            : ['any', 'G_synthesis/64x64/Conv0_up/mod_bias'          ],
            'synthesis_network.blocks.3.conv0_up.noise.noise_scaler'        : ['uns', 'G_synthesis/64x64/Conv0_up/noise_strength'    ],
            'synthesis_network.blocks.3.conv0_up.noise.const_noise'         : ['any', 'G_synthesis/noise7'                           ],
            'synthesis_network.blocks.3.conv0_up.bias.bias'                 : ['any', 'G_synthesis/64x64/Conv0_up/bias'              ],
            'synthesis_network.blocks.3.conv1.conv.weight'                  : ['con', 'G_synthesis/64x64/Conv1/weight'               ],
            'synthesis_network.blocks.3.conv1.conv.fc.weight'               : ['fc_', 'G_synthesis/64x64/Conv1/mod_weight'           ],
            'synthesis_network.blocks.3.conv1.conv.bias.bias'               : ['any', 'G_synthesis/64x64/Conv1/mod_bias'             ],
            'synthesis_network.blocks.3.conv1.noise.noise_scaler'           : ['uns', 'G_synthesis/64x64/Conv1/noise_strength'       ],
            'synthesis_network.blocks.3.conv1.noise.const_noise'            : ['any', 'G_synthesis/noise8'                           ],
            'synthesis_network.blocks.3.conv1.bias.bias'                    : ['any', 'G_synthesis/64x64/Conv1/bias'                 ],

            'synthesis_network.blocks.4.conv0_up.conv.weight'               : ['mTc', 'G_synthesis/128x128/Conv0_up/weight'          ],
            'synthesis_network.blocks.4.conv0_up.conv.fc.weight'            : ['fc_', 'G_synthesis/128x128/Conv0_up/mod_weight'      ],
            'synthesis_network.blocks.4.conv0_up.conv.bias.bias'            : ['any', 'G_synthesis/128x128/Conv0_up/mod_bias'        ],
            'synthesis_network.blocks.4.conv0_up.noise.noise_scaler'        : ['uns', 'G_synthesis/128x128/Conv0_up/noise_strength'  ],
            'synthesis_network.blocks.4.conv0_up.noise.const_noise'         : ['any', 'G_synthesis/noise9'                           ],
            'synthesis_network.blocks.4.conv0_up.bias.bias'                 : ['any', 'G_synthesis/128x128/Conv0_up/bias'            ],
            'synthesis_network.blocks.4.conv1.conv.weight'                  : ['con', 'G_synthesis/128x128/Conv1/weight'             ],
            'synthesis_network.blocks.4.conv1.conv.fc.weight'               : ['fc_', 'G_synthesis/128x128/Conv1/mod_weight'         ],
            'synthesis_network.blocks.4.conv1.conv.bias.bias'               : ['any', 'G_synthesis/128x128/Conv1/mod_bias'           ],
            'synthesis_network.blocks.4.conv1.noise.noise_scaler'           : ['uns', 'G_synthesis/128x128/Conv1/noise_strength'     ],
            'synthesis_network.blocks.4.conv1.noise.const_noise'            : ['any', 'G_synthesis/noise10'                          ],
            'synthesis_network.blocks.4.conv1.bias.bias'                    : ['any', 'G_synthesis/128x128/Conv1/bias'               ],


            'synthesis_network.blocks.5.conv0_up.conv.weight'               : ['mTc', 'G_synthesis/256x256/Conv0_up/weight'          ],
            'synthesis_network.blocks.5.conv0_up.conv.fc.weight'            : ['fc_', 'G_synthesis/256x256/Conv0_up/mod_weight'      ],
            'synthesis_network.blocks.5.conv0_up.conv.bias.bias'            : ['any', 'G_synthesis/256x256/Conv0_up/mod_bias'        ],
            'synthesis_network.blocks.5.conv0_up.noise.noise_scaler'        : ['uns', 'G_synthesis/256x256/Conv0_up/noise_strength'  ],
            'synthesis_network.blocks.5.conv0_up.noise.const_noise'         : ['any', 'G_synthesis/noise11'                          ],
            'synthesis_network.blocks.5.conv0_up.bias.bias'                 : ['any', 'G_synthesis/256x256/Conv0_up/bias'            ],
            'synthesis_network.blocks.5.conv1.conv.weight'                  : ['con', 'G_synthesis/256x256/Conv1/weight'             ],
            'synthesis_network.blocks.5.conv1.conv.fc.weight'               : ['fc_', 'G_synthesis/256x256/Conv1/mod_weight'         ],
            'synthesis_network.blocks.5.conv1.conv.bias.bias'               : ['any', 'G_synthesis/256x256/Conv1/mod_bias'           ],
            'synthesis_network.blocks.5.conv1.noise.noise_scaler'           : ['uns', 'G_synthesis/256x256/Conv1/noise_strength'     ],
            'synthesis_network.blocks.5.conv1.noise.const_noise'            : ['any', 'G_synthesis/noise12'                          ],
            'synthesis_network.blocks.5.conv1.bias.bias'                    : ['any', 'G_synthesis/256x256/Conv1/bias'               ],

            'synthesis_network.blocks.6.conv0_up.conv.weight'               : ['mTc', 'G_synthesis/512x512/Conv0_up/weight'          ],
            'synthesis_network.blocks.6.conv0_up.conv.fc.weight'            : ['fc_', 'G_synthesis/512x512/Conv0_up/mod_weight'      ],
            'synthesis_network.blocks.6.conv0_up.conv.bias.bias'            : ['any', 'G_synthesis/512x512/Conv0_up/mod_bias'        ],
            'synthesis_network.blocks.6.conv0_up.noise.noise_scaler'        : ['uns', 'G_synthesis/512x512/Conv0_up/noise_strength'  ],
            'synthesis_network.blocks.6.conv0_up.noise.const_noise'         : ['any', 'G_synthesis/noise13'                          ],
            'synthesis_network.blocks.6.conv0_up.bias.bias'                 : ['any', 'G_synthesis/512x512/Conv0_up/bias'            ],
            'synthesis_network.blocks.6.conv1.conv.weight'                  : ['con', 'G_synthesis/512x512/Conv1/weight'             ],
            'synthesis_network.blocks.6.conv1.conv.fc.weight'               : ['fc_', 'G_synthesis/512x512/Conv1/mod_weight'         ],
            'synthesis_network.blocks.6.conv1.conv.bias.bias'               : ['any', 'G_synthesis/512x512/Conv1/mod_bias'           ],
            'synthesis_network.blocks.6.conv1.noise.noise_scaler'           : ['uns', 'G_synthesis/512x512/Conv1/noise_strength'     ],
            'synthesis_network.blocks.6.conv1.noise.const_noise'            : ['any', 'G_synthesis/noise14'                          ],
            'synthesis_network.blocks.6.conv1.bias.bias'                    : ['any', 'G_synthesis/512x512/Conv1/bias'               ],

            'synthesis_network.blocks.7.conv0_up.conv.weight'               : ['mTc', 'G_synthesis/1024x1024/Conv0_up/weight'        ],
            'synthesis_network.blocks.7.conv0_up.conv.fc.weight'            : ['fc_', 'G_synthesis/1024x1024/Conv0_up/mod_weight'    ],
            'synthesis_network.blocks.7.conv0_up.conv.bias.bias'            : ['any', 'G_synthesis/1024x1024/Conv0_up/mod_bias'      ],
            'synthesis_network.blocks.7.conv0_up.noise.noise_scaler'        : ['uns', 'G_synthesis/1024x1024/Conv0_up/noise_strength'],
            'synthesis_network.blocks.7.conv0_up.noise.const_noise'         : ['any', 'G_synthesis/noise15'                          ],
            'synthesis_network.blocks.7.conv0_up.bias.bias'                 : ['any', 'G_synthesis/1024x1024/Conv0_up/bias'          ],
            'synthesis_network.blocks.7.conv1.conv.weight'                  : ['con', 'G_synthesis/1024x1024/Conv1/weight'           ],
            'synthesis_network.blocks.7.conv1.conv.fc.weight'               : ['fc_', 'G_synthesis/1024x1024/Conv1/mod_weight'       ],
            'synthesis_network.blocks.7.conv1.conv.bias.bias'               : ['any', 'G_synthesis/1024x1024/Conv1/mod_bias'         ],
            'synthesis_network.blocks.7.conv1.noise.noise_scaler'           : ['uns', 'G_synthesis/1024x1024/Conv1/noise_strength'   ],
            'synthesis_network.blocks.7.conv1.noise.const_noise'            : ['any', 'G_synthesis/noise16'                          ],
            'synthesis_network.blocks.7.conv1.bias.bias'                    : ['any', 'G_synthesis/1024x1024/Conv1/bias'             ],


            'synthesis_network.init_block.to_rgb.conv.weight'               : ['con', 'G_synthesis/4x4/ToRGB/weight'                 ],
            'synthesis_network.init_block.to_rgb.conv.fc.weight'            : ['fc_', 'G_synthesis/4x4/ToRGB/mod_weight'             ],
            'synthesis_network.init_block.to_rgb.conv.bias.bias'            : ['any', 'G_synthesis/4x4/ToRGB/mod_bias'               ],
            'synthesis_network.init_block.to_rgb.bias.bias'                 : ['any', 'G_synthesis/4x4/ToRGB/bias'                   ],

            'synthesis_network.blocks.0.to_rgb.conv.weight'                 : ['con', 'G_synthesis/8x8/ToRGB/weight'                 ],
            'synthesis_network.blocks.0.to_rgb.conv.fc.weight'              : ['fc_', 'G_synthesis/8x8/ToRGB/mod_weight'             ],
            'synthesis_network.blocks.0.to_rgb.conv.bias.bias'              : ['any', 'G_synthesis/8x8/ToRGB/mod_bias'               ],
            'synthesis_network.blocks.0.to_rgb.bias.bias'                   : ['any', 'G_synthesis/8x8/ToRGB/bias'                   ],

            'synthesis_network.blocks.1.to_rgb.conv.weight'                 : ['con', 'G_synthesis/16x16/ToRGB/weight'               ],
            'synthesis_network.blocks.1.to_rgb.conv.fc.weight'              : ['fc_', 'G_synthesis/16x16/ToRGB/mod_weight'           ],
            'synthesis_network.blocks.1.to_rgb.conv.bias.bias'              : ['any', 'G_synthesis/16x16/ToRGB/mod_bias'             ],
            'synthesis_network.blocks.1.to_rgb.bias.bias'                   : ['any', 'G_synthesis/16x16/ToRGB/bias'                 ],

            'synthesis_network.blocks.2.to_rgb.conv.weight'                 : ['con', 'G_synthesis/32x32/ToRGB/weight'               ],
            'synthesis_network.blocks.2.to_rgb.conv.fc.weight'              : ['fc_', 'G_synthesis/32x32/ToRGB/mod_weight'           ],
            'synthesis_network.blocks.2.to_rgb.conv.bias.bias'              : ['any', 'G_synthesis/32x32/ToRGB/mod_bias'             ],
            'synthesis_network.blocks.2.to_rgb.bias.bias'                   : ['any', 'G_synthesis/32x32/ToRGB/bias'                 ],

            'synthesis_network.blocks.3.to_rgb.conv.weight'                 : ['con', 'G_synthesis/64x64/ToRGB/weight'               ],
            'synthesis_network.blocks.3.to_rgb.conv.fc.weight'              : ['fc_', 'G_synthesis/64x64/ToRGB/mod_weight'           ],
            'synthesis_network.blocks.3.to_rgb.conv.bias.bias'              : ['any', 'G_synthesis/64x64/ToRGB/mod_bias'             ],
            'synthesis_network.blocks.3.to_rgb.bias.bias'                   : ['any', 'G_synthesis/64x64/ToRGB/bias'                 ],

            'synthesis_network.blocks.4.to_rgb.conv.weight'                 : ['con', 'G_synthesis/128x128/ToRGB/weight'             ],
            'synthesis_network.blocks.4.to_rgb.conv.fc.weight'              : ['fc_', 'G_synthesis/128x128/ToRGB/mod_weight'         ],
            'synthesis_network.blocks.4.to_rgb.conv.bias.bias'              : ['any', 'G_synthesis/128x128/ToRGB/mod_bias'           ],
            'synthesis_network.blocks.4.to_rgb.bias.bias'                   : ['any', 'G_synthesis/128x128/ToRGB/bias'               ],

            'synthesis_network.blocks.5.to_rgb.conv.weight'                 : ['con', 'G_synthesis/256x256/ToRGB/weight'             ],
            'synthesis_network.blocks.5.to_rgb.conv.fc.weight'              : ['fc_', 'G_synthesis/256x256/ToRGB/mod_weight'         ],
            'synthesis_network.blocks.5.to_rgb.conv.bias.bias'              : ['any', 'G_synthesis/256x256/ToRGB/mod_bias'           ],
            'synthesis_network.blocks.5.to_rgb.bias.bias'                   : ['any', 'G_synthesis/256x256/ToRGB/bias'               ],

            'synthesis_network.blocks.6.to_rgb.conv.weight'                 : ['con', 'G_synthesis/512x512/ToRGB/weight'             ],
            'synthesis_network.blocks.6.to_rgb.conv.fc.weight'              : ['fc_', 'G_synthesis/512x512/ToRGB/mod_weight'         ],
            'synthesis_network.blocks.6.to_rgb.conv.bias.bias'              : ['any', 'G_synthesis/512x512/ToRGB/mod_bias'           ],
            'synthesis_network.blocks.6.to_rgb.bias.bias'                   : ['any', 'G_synthesis/512x512/ToRGB/bias'               ],

            'synthesis_network.blocks.7.to_rgb.conv.weight'                 : ['con', 'G_synthesis/1024x1024/ToRGB/weight'           ],
            'synthesis_network.blocks.7.to_rgb.conv.fc.weight'              : ['fc_', 'G_synthesis/1024x1024/ToRGB/mod_weight'       ],
            'synthesis_network.blocks.7.to_rgb.conv.bias.bias'              : ['any', 'G_synthesis/1024x1024/ToRGB/mod_bias'         ],
            'synthesis_network.blocks.7.to_rgb.bias.bias'                   : ['any', 'G_synthesis/1024x1024/ToRGB/bias'             ]

        }

        self.functions_dict = {
            # EqualizedConv2DTranspose (iC,oC,kH,kW)
            'mTc' : lambda weight: torch.flip(torch.from_numpy(weight.transpose((2,3,0,1))), [2, 3]),
            # Conv2DTranspose (iC,oC,kH,kW)
            'Tco' : lambda weight: torch.from_numpy(weight.transpose((2,3,0,1))), 
            # Conv2D (oC,iC,kH,kW)
            'con' : lambda weight: torch.from_numpy(weight.transpose((3,2,0,1))),
            # FullyConnect (oD, iD)
            'fc_' : lambda weight: torch.from_numpy(weight.transpose((1, 0))),
            # Bias, const_input, noise, v1 noise 
            'any' : lambda weight: torch.from_numpy(weight),
            # Style-Mixing, v2 noise (scalar)
            'uns' : lambda weight: torch.from_numpy(np.array(weight).reshape(1)),
        }

    def convert(self, src_dict):
        new_dict_pt = { k : self.functions_dict[v[0]](src_dict[v[1]]) for k,v in self.name_trans_dict.items()}
        return new_dict_pt

    
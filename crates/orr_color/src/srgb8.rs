//! 8-bit sRGB codes: [`srgb8_encode`] gives the code nearest 255 times the sRGB encoding of a
//! linear value, and [`srgb8_decode`] the f32 nearest a code's linear value, as
//! [`srgb_encode`](crate::srgb_encode) and [`srgb_decode`](crate::srgb_decode) define them in real
//! numbers. No platform's `powf` rounds correctly, so tables stand in for it.

/// Bits below the point of [`HALF_CODES`] and [`LINEAR`].
const POINT: u32 = 127;

/// Per code from 1, the linear value whose encoding is a half code below it, at [`POINT`] bits and
/// rounded down. A value encodes to the code or above where it is above this, and none lies on a
/// point of this grid. The test `tables_are_the_exact_definition` derives them.
#[rustfmt::skip]
const HALF_CODES: [u128; 255] = [
    0x0004_f915_9df0_5dbd_a447_8c80_9d25_77e5, 0x000e_eb40_d9d1_1938_ecd6_a581_d770_67af,
    0x0018_dd6c_15b1_d4b4_3565_be83_11bb_577a, 0x0022_cf97_5192_902f_7df4_d784_4c06_4744,
    0x002c_c1c2_8d73_4baa_c683_f085_8651_370f, 0x0036_b3ed_c954_0726_0f13_0986_c09c_26d9,
    0x0040_a619_0534_c2a1_57a2_2287_fae7_16a4, 0x004a_9844_4115_7e1c_a031_3b89_3532_066f,
    0x0054_8a6f_7cf6_3997_e8c0_548a_6f7c_f639, 0x005e_7c9a_b8d6_f513_314f_6d8b_a9c7_e604,
    0x0068_7968_0ea2_92d5_a508_2fe9_a315_1584, 0x0072_fdcd_11df_839d_9e95_e80a_931d_3fe7,
    0x007e_1a01_6424_4b04_0a23_8a8e_9e84_ee23, 0x0089_d05f_1f4c_56a6_b99c_5f68_1ef7_a2c7,
    0x0096_2332_e35e_2c55_be45_4fdc_6088_471a, 0x00a3_14bc_9d1c_cd80_a6e9_8a5b_4e86_aa62,
    0x00b0_a730_3aaa_0ac0_3ad3_7758_c0a7_3e8c, 0x00be_dcb6_5064_cec4_d789_29d6_33b8_ae99,
    0x00cd_b76c_afdb_6dd3_690e_98df_d6c5_6d1c, 0x00dd_3966_f265_ae4c_6a1c_87fd_7377_32f3,
    0x00ed_64ae_f8c1_94b4_8ec5_ac42_1a2a_dd8b, 0x00fe_3b45_60de_b998_0d95_1235_68fe_a42d,
    0x010f_bf21_f2dc_488a_aa78_0f59_5f9b_31bc, 0x0121_f234_061c_55bb_38eb_d7ee_05e8_8ab1,
    0x0134_d662_df32_ddb9_5dde_d5ae_c60a_685d, 0x0148_6d8e_075e_9d41_3a53_c072_6146_1595,
    0x015c_b98d_9e25_45ec_d9c0_e62c_d55b_6a75, 0x0171_bc32_a59a_d45f_068f_f822_c9b4_c1ce,
    0x0187_7747_49cc_7c48_60ac_9607_8be3_7a97, 0x019d_ec8f_23ba_5bf0_b1da_5f53_89f6_2257,
    0x01b5_1dc7_783f_a29c_c4fa_3a83_23f2_fb32, 0x01cd_0ca7_733e_c83a_ae8c_cddd_8792_56b6,
    0x01e5_bae0_5f5e_a991_f8d1_f8b7_c3d2_4007, 0x01ff_2a1d_da9d_a465_efdd_0e03_e469_4d43,
    0x0219_5c06_07f9_03ab_c211_72e8_7ec2_23d8, 0x0234_5239_be61_0c80_325d_1147_6711_9234,
    0x0250_0e54_b52c_ad54_79dd_e19e_b782_5347, 0x026c_91ed_ae3b_1a79_1b42_785d_a0a7_91c0,
    0x0289_de96_9ded_71e0_c1a5_e53d_e16f_9394, 0x02a7_f5dc_d11e_c2b8_7a78_6894_ed44_2de7,
    0x02c6_d949_113d_7dc2_63ae_35a2_c0e7_7a9e, 0x02e6_8a5f_c6a6_54c6_0c26_d702_976e_9acc,
    0x0307_0aa1_195d_df2a_ab1c_1801_67b3_9841, 0x0328_5b89_1043_eea5_c2b7_b55f_de5b_40a1,
    0x034a_7e8f_aed9_50cd_2982_1436_21d8_a34a, 0x036d_7529_11ae_c254_a650_ad87_16f4_446f,
    0x0391_40c5_8992_10ba_e356_c814_62d4_7142, 0x03b5_e2d1_b58c_c9da_bb0f_6290_2ec8_0ca4,
    0x03db_5cb6_9bc6_6199_a62d_d316_4c13_14d2, 0x0401_afd9_c15a_60b1_4561_0e6b_dd32_d1f6,
    0x0428_dd9d_4131_f8c5_9395_00e4_1956_e3c9, 0x0450_e75f_e1ef_3e75_34a8_a153_0095_cfe8,
    0x0479_ce7d_2af7_470e_16fc_1234_cdfd_a016, 0x04a3_944d_78a7_7a98_f355_2543_ce65_d1e8,
    0x04ce_3a26_0fc1_91d4_720a_2ec1_fb9c_43c3, 0x04f9_c159_3018_f08d_b801_e3fe_839f_2c50,
    0x0526_2b36_268b_55cf_68a1_c9b3_3fae_e478, 0x0553_7909_5e4e_3319_48f0_f078_4635_938f,
    0x0581_ac1c_7199_60db_4cd5_3836_c271_91d7, 0x05b0_c5b6_39b7_578d_b102_ae1c_d964_3f99,
    0x05e0_c71a_de82_90dc_4f7f_e818_11ef_7a8b, 0x0611_b18b_e557_3a79_f318_24a5_5786_63d7,
    0x0643_8648_3f7f_f36e_fbae_fc03_d660_048d, 0x0676_468c_5823_e449_c1b1_991d_9765_d0be,
    0x06a9_f392_21bc_21c7_8af0_74ca_76a8_1b0e, 0x06de_8e91_2315_efb5_2fea_8806_6bfe_daaa,
    0x0714_18be_83e7_2651_5a60_333a_a22b_d791, 0x074a_934d_18f9_aee0_5c96_ee6d_1eb4_cdaf,
    0x0781_ff6d_6ff2_c3e8_8b46_7276_1b7d_3834, 0x07ba_5e4d_daba_5e46_ab1d_c6a7_951f_366d,
    0x07f3_b11a_7a86_f98f_8128_d31e_900e_cbca, 0x082d_f8fd_4a91_9faa_2cc0_2752_6e9f_29b5,
    0x0869_371e_2a75_f4e3_1cc0_600d_3dee_7bdf, 0x08a5_6ca2_e841_ca9d_113e_67d2_5251_c1da,
    0x08e2_9aaf_4a37_8ef7_39d4_2e7b_05f7_4920, 0x0920_c265_1846_c2fd_ad33_4f40_62dd_7eb4,
    0x095f_e4e4_253d_770a_2d43_a7b8_be06_d01f, 0x09a0_034a_57b5_a5a9_5108_f049_2c1b_752d,
    0x09e1_1eb3_b2c1_2073_1a1c_a07b_dd30_8480, 0x0a23_383a_5e56_9fa1_0d48_5052_5411_f1f4,
    0x0a66_50f6_af82_649e_2302_49e9_ae09_0bf6, 0x0aaa_69ff_305c_c130_a572_8738_4a36_a3ac,
    0x0aef_8468_a7c8_b810_b345_3f1a_b814_8aba, 0x0b35_a146_20fc_d09f_bd66_62cf_fd94_488c,
    0x0b7c_c1a8_f2d8_1de8_8f3d_dfa5_96e9_89d3, 0x0bc4_e6a0_c705_60f9_2b2a_d14e_0764_4000,
    0x0c0e_113b_a0ee_17ee_fb81_9e7f_9dbc_b85d, 0x0c58_4285_e47f_359b_6e9f_dd89_ec86_8f5e,
    0x0ca3_7b8a_5cc1_2965_ac42_29a4_42d4_3c5f, 0x0cef_bd52_4244_cbe9_e514_f65c_8f7f_934b,
    0x0d3d_08e5_4166_b2bd_7ba0_3984_34c5_b06a, 0x0d8b_5f49_806a_5c9f_f690_120c_ba82_6381,
    0x0dda_c183_a56e_972b_926c_af5d_c0fe_f828, 0x0e2b_3096_dc3c_6fbe_c7f8_501c_8169_8dda,
    0x0e7c_ad84_dbf1_f2cc_a2af_43bd_ba80_092c, 0x0ecf_394d_ec89_eef7_b341_4c27_6dff_ed4e,
    0x0f22_d4f0_ec41_e447_08d1_49bb_d768_9eff, 0x0f77_816b_54df_3b5a_fdbe_c416_5edc_a91d,
    0x0fcd_3fb9_40d4_d4be_9ddc_247d_bc43_7e88, 0x1024_10d5_7049_f540_8164_4828_d2bf_0473,
    0x107b_f5b9_4e03_899a_526c_2a4b_82be_eca6, 0x10d4_ef5c_f430_b190_91c2_0cd8_fa95_62b2,
    0x112e_feb7_311b_7914_dfce_d38d_0872_6789, 0x118a_24bd_8bbe_9ccc_db35_bdcb_ac61_bce4,
    0x11e6_6264_4841_2ea7_d699_6088_629a_b452, 0x1243_b89e_6c58_e6df_ed42_3b09_2784_d90a,
    0x12a2_285d_c393_e5d9_76a7_8dd7_eb20_ad0f, 0x1301_b292_e38a_a3c8_fe07_ea26_5654_232f,
    0x1362_582d_2ffa_c3d6_8933_4663_09fb_5f98, 0x13c4_1a1a_decb_7998_46e1_cca1_1552_801e,
    0x1426_f948_fbfc_2932_24c3_ebfb_d308_2e2c, 0x148a_f6a3_6d7d_e427_0ab3_4c87_8e7a_d72a,
    0x14f0_1314_f6f8_5eef_8874_afc5_f9c7_8e2d, 0x1556_4f87_3d7a_f5b4_e1b8_d745_b5ab_a109,
    0x15bd_ace2_cb1a_511b_05b6_8626_830d_f2d5, 0x1626_2c0f_127b_36cc_cff5_ebe5_3e3c_ec3a,
    0x168f_cdf2_724b_0c80_cc2c_f231_9553_6721, 0x16fa_9372_38a6_8f66_abcf_951f_d3eb_ae67,
    0x1766_7d72_a66f_3d5b_e191_2c8d_baa4_1966, 0x17d3_8cd6_f28f_e8e5_cee5_aab0_ce15_2761,
    0x1841_c281_4d30_ecbf_215f_1448_bb7b_7749, 0x18b1_1f52_e2dc_6fc2_1537_a7a0_1a86_bae2,
    0x1921_a42b_df93_2620_1cbe_a114_504b_d183, 0x1993_51eb_71d1_f926_d67c_2d5a_141e_66f2,
    0x1a06_296f_cd88_fb48_5003_8215_8146_de3b, 0x1a7a_2b96_2f04_0ab7_7786_2afd_5b3c_a4c0,
    0x1aef_593a_ddc5_81a7_923b_ad98_3f33_5215, 0x1b65_b339_2f53_501c_e9da_b944_d671_f6bd,
    0x1bdd_3a6b_89f6_d83c_16da_02dd_0e69_2a38, 0x1c55_efab_676f_e322_dc97_4260_5cad_fb96,
    0x1ccf_d3d1_579b_0190_10ab_b1e9_b3d0_bd7c, 0x1d4a_e7b5_030b_a8f6_2b6b_8086_42a0_85fd,
    0x1dc7_2c2d_2d9a_5b04_a278_5c2f_9eb3_bbe4, 0x1e44_a20f_b8e7_233d_dde3_3782_9647_c43a,
    0x1ec3_4a31_a6d0_b3d1_52dc_bd09_303c_e511, 0x1f43_2567_1be0_68aa_0192_236b_8848_c895,
    0x1fc4_3483_61ab_7570_39c2_3962_db38_2d7c, 0x2046_7858_e929_8121_2b44_7143_4fa7_fd7d,
    0x20c9_f1b9_4d00_efd6_7a13_3536_af2f_35c7, 0x214e_a175_53c9_1962_f11a_89e9_e52d_8d3b,
    0x21d4_885c_f242_a986_afdd_ff94_9e04_f33b, 0x225b_a73f_4d86_64ac_0343_864a_0cc0_4f5e,
    0x22e3_feea_bd2a_8a5d_c1a1_8349_54b2_8930, 0x236d_902c_cd5f_0cf7_c71e_664b_9117_3b9e,
    0x23f8_5bd2_4100_d471_63df_c9d6_1bad_0c4f, 0x2484_62a7_13a4_408f_8c34_3fa4_8e82_fdba,
    0x2511_a576_7b97_1d48_c673_4b67_eebd_8d69, 0x25a0_250a_ebda_3aae_94e4_68cf_366f_c6bf,
    0x262f_e22e_1612_d846_e4cf_3682_bf5e_fb0e, 0x26c0_dda8_ec74_1263_56cf_c1a2_3bed_aaab,
    0x2753_1843_a3a0_7eb3_7df0_61e0_fa06_6176, 0x27e6_92c5_b484_2409_ff29_5567_3111_1bf0,
    0x287b_4df5_de26_f812_53bc_70a3_a28b_5908, 0x2911_4a9a_2778_0c85_65d8_8148_3b3c_49b6,
    0x29a8_8977_e111_9445_e48d_7661_234b_7f78, 0x2a41_0b53_a6f5_e7b1_9bc6_ef63_b943_6785,
    0x2ada_d0f1_6245_ae62_f4c3_f10f_83d9_d091, 0x2b75_db14_4aef_5394_a5c0_3011_5645_0b9b,
    0x2c12_2a7e_e957_e959_2acf_bf68_2b1e_d3a4, 0x2caf_bff3_17fd_9ddf_90cf_c965_ff6d_2175,
    0x2d4e_9c32_0513_e50f_140f_6a99_3231_c9e6, 0x2dee_bffc_3419_77da_c8ce_9c60_2ddb_8c00,
    0x2e90_2c11_7f68_49cd_a927_d97b_8e39_2c67, 0x2f32_e131_19bf_9475_b2e8_7086_be34_520f,
    0x2fd6_e019_8fc8_1781_f3d2_e0f3_bb30_97d2, 0x307c_2988_c992_ab9b_3a42_94dd_b8a2_edfc,
    0x3122_be3c_0c11_4539_5c6e_1207_890c_9f97, 0x31ca_9eef_fa8a_83f7_6d80_5b3f_ab52_2574,
    0x3273_cc60_9807_ea2f_8be6_2c6e_c06c_567a, 0x331e_4749_48be_d7f3_0052_5c6d_0795_11c7,
    0x33ca_1064_d374_63c4_f44b_4740_dc68_8c17, 0x3477_286d_62dc_2ad4_e6de_6bec_af49_012c,
    0x3525_901c_86f2_31d2_05c9_879b_88d3_3f25, 0x35d5_482b_364f_eed1_8268_4c78_455b_120e,
    0x3686_5151_cf7c_9427_af4f_8b8d_0bbe_d688, 0x3738_ac48_1a38_b37e_02cd_8b77_e138_90c5,
    0x37ec_59c5_48c5_4fde_da60_949f_9954_4092, 0x38a1_5a7f_f926_74e2_e413_b191_cd7a_4a06,
    0x3957_af2e_3661_68a3_3ea4_6ce4_f3dc_ef9c, 0x3a0f_5885_79b6_8d8e_8bea_04d1_7353_124c,
    0x3ac8_573a_abd7_08be_372d_e639_238c_a3b5, 0x3b82_ac02_2616_40eb_f15c_7191_2492_6e70,
    0x3c3e_578f_b397_49ad_beda_2369_4974_b105, 0x3cfb_5a96_9276_4e28_c74b_69b0_3f6f_f2a8,
    0x3db9_b5c9_74ee_0df8_48df_01d2_37a4_da30, 0x3e79_69da_8279_7e96_81f3_bdd8_419b_664a,
    0x3f3a_777b_58f1_a328_f9a4_1631_41a3_508e, 0x3ffc_df5d_0da7_ac28_2f4a_0397_0687_0239,
    0x40c0_a230_2e7b_6ff4_48db_b71f_b389_b03a, 0x4185_c0a4_c2ee_4d04_bf7b_d1f3_bb07_7ef3,
    0x424c_3b6a_4d32_8600_310f_a849_5705_c2fd, 0x4314_132f_cb37_27ab_4e57_dec2_2bb1_dd48,
    0x43dd_48a3_b7b0_8842_40ef_d760_2cbb_78dc, 0x44a7_dc74_0b1d_6f76_c69a_f76f_a23a_f5aa,
    0x4573_cf4e_3cc8_f5f6_8b06_be96_2d96_ab6e, 0x4641_21df_43c9_2b08_eb07_8377_930d_489d,
    0x470f_d4d3_97fa_8e81_316b_d2e8_b13b_70ee, 0x47df_e8d7_32f8_6cf2_7397_9040_547f_7139,
    0x48b1_5e95_9112_2bc4_699b_e3ee_64fc_e001, 0x4984_36b9_b23d_927c_e184_e869_8dca_92a6,
    0x4a58_71ee_1b06_1e47_afc5_d26c_f4e1_645d, 0x4b2e_10dc_d579_6c7f_2f48_927d_aefb_0a4b,
    0x4c05_142f_7210_c8b1_737d_ce6b_8c4b_6951, 0x4cdd_7c8f_0897_ea5a_2f35_b02e_58f1_1a6b,
    0x4db7_4aa4_3910_ee46_f5e5_063f_0158_a013, 0x4e92_7f17_2c95_975a_d49e_8511_f1d6_1855,
    0x4f6f_1a8f_9635_e227_3f48_f1f1_fe40_7e36, 0x504d_1db4_b3d3_f692_e8c0_898c_3692_c3fb,
    0x512c_892d_4efd_828b_4554_0dd8_1cb4_ed8b, 0x520d_5d9f_bdc2_8883_2290_469a_4601_9a1b,
    0x52ef_9bb1_e389_ac47_e21a_ad29_d5bb_e785, 0x53d3_4409_31e2_087f_7464_54f6_45a3_5076,
    0x54b8_574a_a952_96eb_208f_8ad0_bbdf_5cc8, 0x559e_d61a_da27_3556_6ea7_e262_cf5c_6cce,
    0x5686_c11d_e53b_50e6_1e48_e35a_d3bb_c3b6, 0x5770_18f7_7cc2_4148_ec5b_f094_dab6_1e0f,
    0x585a_de4a_e50d_5d1a_fd5d_223f_7cd9_472b, 0x5947_11ba_f54f_d09d_0592_6b16_d8db_c6e4,
    0x5a34_b3ea_1860_3fb1_b516_5c25_5abf_f4d0, 0x5b23_c57a_4d78_3be6_7a17_4cd9_9a4f_e417,
    0x5c14_470d_28f1_9720_500a_3bcf_80ac_1e13, 0x5d06_3943_d501_9b5a_06e7_feaa_b4f1_35d1,
    0x5df9_9cbf_1272_2fb7_2a36_5921_9205_8578, 0x5eee_721f_3958_f305_6d37_731a_8d1c_174f,
    0x5fe4_ba04_39cc_539e_38f7_c958_3de2_668e, 0x60dc_750d_9c96_ac72_a326_9661_0689_c4bf,
    0x61d5_a3da_83e7_6ee5_aaf8_c5ae_f74c_8241, 0x62d0_4709_ac02_60f3_154b_2fc4_0673_993f,
    0x63cc_5f39_6bec_f6fc_9c83_0922_28a0_ec49, 0x64c9_ed07_b619_d073_5d32_a1a2_9751_259f,
    0x65c8_f112_1912_5e70_7065_2e03_a5b1_c5a0, 0x66c9_6bf5_c01e_bb2d_79ed_da03_b97a_973e,
    0x67cb_5e4f_73eb_b92c_8ea1_c402_934f_12c5, 0x68ce_c8bb_9b2f_30be_36a2_9974_c355_846b,
    0x69d3_abd6_3b4a_9274_6d9b_8336_ccdd_9687, 0x6ada_083a_f8eb_c4f2_58ff_f7bd_cd13_592b,
    0x6be1_de85_18ac_5469_f617_6355_899b_2ad0, 0x6ceb_2f4f_7fae_f9fb_3a2b_02ae_aca9_d326,
    0x6df5_fb34_b43b_810b_01eb_34a7_06a2_7dfc, 0x6f02_42ce_de59_108b_b5a0_e281_42d8_a31f,
    0x7010_06b7_c866_de15_b1aa_b39e_ddc5_ab3e, 0x711f_4788_dfb3_5092_49ea_083d_69ba_de81,
    0x7230_05db_3511_9821_9f0e_af35_20dc_df21, 0x7342_4247_7d6d_c0c9_722e_6f6d_1637_5bed,
    0x7455_fd66_125f_4561_a9f1_37fe_2766_c88c, 0x756b_37ce_f2ba_281a_511b_72b1_308d_a419,
    0x7681_f219_c31e_95de_74c8_dded_dad1_a4f3, 0x779a_2cdd_ce87_19bf_6db9_dbef_8211_0928,
    0x78b3_e8b2_06d5_657c_c843_d3d7_b050_be06, 0x79cf_262d_055d_b426_2674_e8b9_d0ca_f1f9,
    0x7aeb_e5e5_0b70_cbbe_1c96_35ed_bd49_21d3, 0x7c0a_2870_02e4_a2af_264f_e5b9_be45_de13,
    0x7d29_ee63_7e9b_adce_6740_b5c5_f55a_6d4d, 0x7e4b_3854_bb0a_db92_f1ed_db51_4c79_7ae8,
    0x7f6e_06d8_9ebe_4113_c8ad_31eb_f6ec_488e,
];

/// Per code, its linear value at [`POINT`] bits, rounded down.
#[rustfmt::skip]
const LINEAR: [u128; 256] = [
    0x0000_0000_0000_0000_0000_0000_0000_0000, 0x0009_f22b_3be0_bb7b_488f_1901_3a4a_efca,
    0x0013_e456_77c1_76f6_911e_3202_7495_df95, 0x001d_d681_b3a2_3271_d9ad_4b03_aee0_cf5f,
    0x0027_c8ac_ef82_eded_223c_6404_e92b_bf2a, 0x0031_bad8_2b63_a968_6acb_7d06_2376_aef4,
    0x003b_ad03_6744_64e3_b35a_9607_5dc1_9ebf, 0x0045_9f2e_a325_205e_fbe9_af08_980c_8e89,
    0x004f_9159_df05_dbda_4478_c809_d257_7e54, 0x0059_8385_1ae6_9755_8d07_e10b_0ca2_6e1e,
    0x0063_75b0_56c7_52d0_d596_fa0c_46ed_5de9, 0x006d_a8c6_d496_724e_a46f_8c02_305c_f15c,
    0x0078_78c7_5c32_676b_4fa3_5bc4_55aa_b59c, 0x0083_e1c5_fc2c_1d32_796a_2066_e30a_0bfb,
    0x008f_e615_f76b_2103_2f3e_d8cd_e3f8_e317, 0x009c_87fd_7caf_13b4_a1d9_f64e_378c_fec7,
    0x00a9_c9b6_63de_4938_80fc_211e_ef2f_6751, 0x00b7_ad6e_da86_4894_ce46_6e0c_b0ba_e4a8,
    0x00c6_354a_018d_aafb_d7c3_fe65_07af_f581, 0x00d5_6360_7dcb_9a66_23fc_c6cf_da7e_38ce,
    0x00e5_39c0_fcfb_014b_f197_fa28_c984_57a8, 0x00f5_ba70_b04b_c64f_1d74_03cb_f971_a11a,
    0x0106_e76b_bda9_2b77_eafe_9352_49bb_8cdb, 0x0118_c2a5_a8a8_0412_9730_70fb_94e8_7cbc,
    0x012b_4e09_b3f0_ae26_64ea_134b_4b9a_a317, 0x013e_8b7b_3bde_961c_f768_d3ee_dace_e605,
    0x0152_7cd6_0af8_b855_54d7_30f7_f4fc_c9a1, 0x0167_23ee_a8d3_7063_2268_f885_c3ee_81de,
    0x017c_8292_a3db_6b10_08f0_edbc_982d_acfc, 0x0192_9a88_d67b_51e4_6394_9d0d_96fc_8e19,
    0x01a9_6d91_a801_6ba7_627c_bf65_9e6f_8f21, 0x01c0_fd67_499f_ab42_a0a2_e4fb_a7c2_2cf4,
    0x01d9_4bbd_efd7_40bc_c4c7_6b9f_0354_89cd, 0x01f2_5a44_0898_83cb_cf0b_d511_18cb_a3a5,
    0x020c_2aa2_6e58_d262_fde3_4fcc_63a4_57ad, 0x0226_be7c_9859_9bfd_29da_b513_250e_ed94,
    0x0242_1770_c856_2e39_e5c1_8da7_a7fb_b6d1, 0x025e_3718_35c8_d751_34db_2200_688f_826c,
    0x027b_1f07_36f3_815c_f93b_50ae_a7bf_ff6f, 0x0298_d0cd_67d3_f427_69e3_eaf8_a6d6_0366,
    0x02b7_4df5_cf28_6264_d991_3490_b0e4_c6c7, 0x02d6_9807_01a5_bd8b_6b09_a0b4_a7b8_3f32,
    0x02f6_b083_437e_754d_6ec8_7bc2_cc5c_7cd8, 0x0317_98e8_a855_bb1a_67ee_ab3f_cec1_6dd8,
    0x0339_52b1_31b9_1657_2b32_6f27_2c4a_cea5, 0x035b_df52_ec38_03bc_3678_1fd1_90ea_9910,
    0x037f_4040_0b2f_7af9_0026_d339_79d3_34e4, 0x03a3_76e7_035d_83af_cd45_52f8_9147_1dd5,
    0x03c8_84b2_a44f_7922_e007_0c2e_5721_d6c8, 0x03ee_6b0a_30bc_362a_20ac_4e49_8e79_fa4a,
    0x0415_2b51_75da_1d6f_f91b_09b6_a1bb_90c1, 0x043c_c6e8_e1bf_c92a_604f_7aa7_170d_567c,
    0x0465_3f2d_98dd_1e94_ef55_fb19_3099_89db, 0x048e_9579_8a99_89bd_59dd_8a14_0a1a_e1c5,
    0x04b8_cb23_8523_436c_876e_9f9e_807e_3a8c, 0x04e3_e17f_487a_b30e_73db_46ef_2d96_bf4e,
    0x050f_d9dd_98c4_3f9f_09c4_a80c_3bdc_3dcb, 0x053c_b58c_4fea_3332_ab8c_1cfe_e03b_1760,
    0x056a_75d6_6e97_b33b_5a3b_fc49_7e9a_eb5a, 0x0599_1c04_2c95_3ae4_2961_979a_7a83_8031,
    0x05c8_a95b_088e_7b9b_8f1e_da87_b266_11da, 0x05f9_1f1d_d749_0920_79a9_8cd3_8e92_2578,
    0x062a_7e8c_d252_c12f_6cfd_196e_dc94_0445, 0x065c_c8e5_a62e_726d_dd2e_8c24_9bba_4b06,
    0x068f_ff63_8004_e1a9_77e0_1542_47e7_3c9e, 0x06c4_233f_1adf_ef41_0968_9527_d5ba_77db,
    0x06f9_35ae_cc75_47f7_f59d_23d0_8089_2b74, 0x072f_37e6_9185_bbfd_be65_2bb1_c84f_9d2d,
    0x0766_2b18_19d6_0c18_2450_da3a_83ae_7ec8, 0x079e_1072_d3c5_b72b_6f4c_0001_a409_b4df,
    0x07d6_e923_f788_1167_847d_cb3a_8a62_1f67, 0x0810_b656_9203_b1c9_6841_048e_514c_3f9b,
    0x084b_7933_8f5c_0a10_21fc_bd39_7900_0f97, 0x0887_32e1_c528_c869_f20a_3233_d59d_a3bb,
    0x08c3_e485_fc5e_70b2_b3bd_7479_8854_84ec, 0x0901_8f42_faeb_6cdb_bf6c_d795_1849_0788,
    0x0940_3439_8d0c_a8c4_00da_dad5_c2b3_4481, 0x097f_d488_8e5c_a52c_be40_88f5_6704_1faa,
    0x09c0_714c_f29f_c761_e7c0_0354_e3b5_8a81, 0x0a02_0ba1_ce50_876d_1dc3_3b2f_fd39_095f,
    0x0a44_a4a0_5eed_fd16_2c75_7b1b_c821_9631, 0x0a88_3d60_130f_2c63_3c88_d5cf_cac8_49f7,
    0x0acc_d6f6_923d_549a_5197_1cd8_6c36_cf51, 0x0b12_7277_c497_68d3_a1a6_2097_2979_157f,
    0x0b59_10f5_da40_beda_4408_2fd5_4adb_f36a, 0x0ba0_b381_529c_e84a_7065_ab79_1874_1765,
    0x0be9_5b29_035a_9278_001a_6f12_0f41_ba2d, 0x0c33_08fa_1f4f_3395_1140_1d0f_3187_1791,
    0x0c7d_be00_3d25_36b9_5324_b6fa_bbcf_d52c, 0x0cc9_7b45_5dde_44ba_19bb_f094_151f_9b9c,
    0x0d16_41d1_f32b_3526_d1e9_e0f3_abf9_05b8, 0x0d64_12ac_e59b_2123_32cc_03dc_24e0_61a5,
    0x0db2_eedb_9aa3_0130_38a6_ae6e_b425_add5, 0x0e02_d761_fa7f_2f2f_6b12_ec60_87d8_ad54,
    0x0e53_cd42_75f0_16fb_1533_e6db_68d9_bc46, 0x0ea5_d17e_0bd3_51c4_d724_c2b8_78da_4722,
    0x0ef8_e514_4e9a_5afe_2984_804b_26c1_3c31, 0x0f4d_0903_69a0_00d0_bd96_1cfd_6063_349d,
    0x0fa2_3e48_265d_a614_8c0c_a05d_c593_08d3, 0x0ff8_85dd_f181_6035_10b7_c2fe_09e0_804c,
    0x104f_e0be_dfe5_f08d_7a64_4913_5361_bf6d, 0x10a8_4fe3_b36d_8e63_f21a_42c0_ba54_cbe7,
    0x1101_d443_dfc0_6cce_9288_1ec6_9b19_0874, 0x115c_6ed5_8eef_de69_c576_d238_3a5f_18cf,
    0x11b8_208d_a5fe_efd8_8bc7_b31f_c95a_6029, 0x1214_ea5f_c951_4985_3257_bc24_714a_62cc,
    0x1272_cd3e_6101_20fe_0be2_6226_a127_68a5, 0x12d1_ca1a_9d1c_fa90_41a1_f16f_701a_60b1,
    0x1331_e1e4_79cd_f465_5232_0fe5_f835_d949, 0x1393_158a_c367_4d61_6611_88c4_1735_cfcb,
    0x13f5_65fb_1a5f_d34c_88a4_da50_0e8c_4e1d, 0x1458_d421_f735_dd6a_ab81_78fc_cead_69bb,
    0x14bd_60ea_ae3e_7289_fcc4_81be_9239_e9b2, 0x1523_0d3f_7360_33b8_a538_8797_86f7_33a4,
    0x1589_da09_5dba_9f3e_d0e6_f35c_bdd4_5d42, 0x15f1_c830_6b3a_3a24_8508_1057_19bb_937e,
    0x165a_d89b_841a_2970_fdd5_b023_23b9_1e78, 0x16c5_0c30_7e53_bf70_12eb_1444_9f3a_44f3,
    0x1730_63d4_20fc_7c9e_945f_39a4_102d_30ab, 0x179c_e06a_2792_ff66_02da_e54c_2db9_f741,
    0x180a_82d5_453b_5987_e676_5630_d9b6_d11b, 0x1879_4bf7_27eb_3dfe_f52e_97bb_3ad3_e717,
    0x18e9_3cb0_7b86_762d_ef01_d2ec_5687_3745, 0x195a_55e0_ecec_096f_7087_6e54_5a17_5a94,
    0x19cc_9867_2cf4_7e7c_f8ea_a1f0_6569_df08, 0x1a40_0520_f361_9aac_15e5_73bd_90cc_6b1e,
    0x1ab4_9ceb_01bf_ffad_4adb_e358_28c5_682e, 0x1b2a_60a1_263b_0547_23cc_d3ee_f2ba_14cf,
    0x1ba1_511e_3e63_2977_5d5b_630b_019b_1949, 0x1c19_6f3c_39e7_6e72_9ac5_6083_3a9a_5287,
    0x1c92_bbd4_1d41_fb27_5813_60f6_1583_16a1, 0x1d0d_37be_0458_5031_44a0_98ab_91c6_d9a0,
    0x1d88_e3d1_250f_608d_ad21_314b_c469_ee4f, 0x1e05_c0e3_d1d3_dade_dabb_c5ba_328c_3413,
    0x1e83_cfcb_7c16_eda2_2519_e5d5_42f6_dd6f, 0x1f03_115c_b6bf_ce69_cd59_430f_feb5_a7c6,
    0x1f83_866b_3892_49ef_83ce_5eb2_c501_4973, 0x2005_2fc9_de8a_a0ad_bc7d_a0c5_9f5b_30ff,
    0x2088_0e4a_ae2e_f19b_bcd4_d66a_8e74_8043, 0x210c_22be_d7d6_72a9_b367_51db_d31a_f4dd,
    0x2191_6df6_b8e6_b4ae_568b_203d_15ba_921f, 0x2217_f0c1_de07_2e9d_a9d1_6f41_019a_824b,
    0x229f_abef_054b_4a17_dfbb_5687_e49b_d5b3, 0x2328_a04c_2053_29a7_19aa_f6a7_d5ad_e6b3,
    0x23b2_cea6_5663_5f5a_59ec_061f_fa5d_b6f1, 0x243e_37ca_0673_c8d2_a964_7d76_eac9_bb7f,
    0x24ca_dc82_c935_c44c_9acd_dabd_03f5_cb75, 0x2558_bd9b_7311_f0b3_5eb2_7539_50a7_0af5,
    0x25e7_dbde_161d_a95a_f389_4a4e_3d39_81c4, 0x2678_3814_0408_6c9c_18ca_9126_de53_4498,
    0x2709_d305_d001_5b36_05dc_12b7_5948_c7fc, 0x279c_ad7b_5094_fd0e_ff90_6c21_f9bf_e27b,
    0x2830_c83b_a183_76ad_47d6_acd6_8601_bab2, 0x28c6_240d_258f_598d_17dc_cdf3_c185_ce10,
    0x295c_c1b5_8845_384d_ef50_57db_8ccf_a592, 0x29f4_a1f9_bfbc_2691_1a95_5418_61de_b8ba,
    0x2a8d_c59e_0e4f_4b4c_817a_4bc5_b6dd_f044, 0x2b28_2d66_0450_ab47_385e_8e50_e47c_e923,
    0x2bc3_da14_81b5_5070_9372_3e83_165e_89e8, 0x2c60_cc6b_b7ba_f1c6_4740_9abd_d292_cc1e,
    0x2cff_052d_2a87_3e8a_1f82_b5a2_f8c1_4f7b, 0x2d9e_8519_b2c0_ee9b_b563_9c7c_b9ac_66da,
    0x2e3f_4cf1_7f22_b8e6_047a_d7a8_2dd2_2710, 0x2ee1_5d74_1608_4ff3_80b3_8611_8fe4_d735,
    0x2f84_b760_56f5_83e5_1928_0fe1_a45c_667c, 0x3029_5b74_7c17_a839_21b7_2e4f_19dd_e6b6,
    0x30cf_4a6e_1bc1_5b06_2ea4_09c5_5c4c_4ffc, 0x3176_850a_29e0_ca8b_458d_3e91_8de9_cc96,
    0x321f_0c04_f970_953a_2aa0_e026_044a_57fc, 0x32c8_e01a_3de3_5fa9_c7d8_387d_5e48_1b76,
    0x3374_0205_0c8a_3b2d_8038_5ce1_704c_1813, 0x3420_727f_ddf5_f722_7ed5_1b56_4bee_3810,
    0x34ce_3244_8f53_765d_8704_d6f7_358f_1bb9, 0x357d_420c_63c3_2182_4941_2967_df87_92bc,
    0x362d_a290_05ab_8e70_98f5_ed2d_5f90_5893, 0x36df_5487_8807_745b_e735_1ebd_4f86_f77e,
    0x3792_58aa_67af_038d_f2e4_9dcc_9437_b819, 0x3846_afaf_8c9c_b746_882d_f520_bfba_0bf8,
    0x38fc_5a4d_4b2d_b79f_4806_f20b_5b25_f659, 0x39b3_5939_655d_e0d3_a064_cfb7_be5d_9e00,
    0x3a6b_ad29_0bff_84ca_3a5c_1002_8c66_af24, 0x3b25_56d0_dfee_fb3e_0ac5_4236_c434_eb6a,
    0x3be0_56e4_f342_1469_c302_f1b2_14bd_ee76, 0x3c9c_ae18_ca73_81a0_7a87_2fdb_d355_1e2f,
    0x3d5a_5d1f_5d8a_46c9_f042_1a7a_eaa3_c48c, 0x3e19_64ab_193d_4747_97ed_2d7e_8037_b652,
    0x3ed9_c56d_e013_0059_ae07_3b51_0ff3_cc42, 0x3f9b_8019_0b7d_82b0_ab33_9a10_e125_f73c,
    0x405e_955d_6cf2_bc6f_8a7c_95d9_c9c6_454d, 0x4123_05eb_4f01_247e_54a1_907e_4504_c3d0,
    0x41e8_d272_7660_d7aa_2cf9_98fb_c313_4f56, 0x42af_fba2_2301_37b0_9b4f_dbf2_b240_cb5d,
    0x4378_8229_1113_1bf6_eb19_84e7_5a77_b8c4, 0x4442_66b5_7a0f_a354_1df1_e323_24a3_0b4e,
    0x450d_a9f5_15bb_b5fc_08aa_6e5c_7449_9edf, 0x45da_4c95_1b28_4644_ab73_dfb8_144d_3ea2,
    0x46a8_4f42_41af_5eab_9b71_e978_52e5_b0a2, 0x4777_b2a8_c1ee_0b30_53dd_1ac7_3f88_fd79,
    0x4848_7774_56bb_2bc9_6cca_cd68_512a_70b5, 0x491a_9e50_3e1b_3d6d_f48a_ebd5_bc0c_9ef9,
    0x49ee_27e7_3a31_26e1_6da6_49b7_fee9_3b06, 0x4ac3_14e3_922c_1628_2edd_315c_f643_6e73,
    0x4b99_65ef_1332_7b43_049d_43bd_fe6b_de21, 0x4c71_1bb3_114a_2c8b_ed54_2c78_2c1a_790f,
    0x4d4a_36d8_683d_c0ca_8d64_81ec_c7b1_6cd8, 0x4e24_b807_7c7f_2ad5_766e_e172_0b39_7c14,
    0x4f00_9fe8_3c07_a255_8ab2_2de5_e432_3a45, 0x4fdd_ef22_1f34_e502_9388_ff00_c1a7_5b5b,
    0x50bc_a65c_29a3_db73_8500_4cb1_36ed_c98a, 0x519c_c63c_eb08_ac61_d41a_95f0_7b8a_288f,
    0x527e_4f6a_8004_4905_ade3_2f4f_bbd3_c4a8, 0x5361_428a_92f7_7cf8_b7bc_9a23_b65d_dc98,
    0x5445_a042_5cd3_8bd6_4271_f9e0_5e1d_41e2, 0x552b_6936_a5e8_669a_7704_18d9_b8d5_73c0,
    0x5612_9e0b_c6b0_828d_f3c5_7420_163d_5c43, 0x56fb_3f65_a89a_5b58_884c_ed35_5476_24b7,
    0x57e5_4de7_c6cf_a9a4_3596_28ee_1ac9_dfc8, 0x58d0_ca35_2efa_5789_4423_f60d_0edd_1ecd,
    0x59bd_b4f0_8207_3bcd_1b44_8360_494a_ff97, 0x5aac_0ebb_f4e6_a4cf_821b_881f_766a_c2e6,
    0x5b9b_d839_514a_bbd5_0c86_7ebb_bb8d_044b, 0x5c8d_1209_f663_c931_a342_a1c1_9f85_f5ea,
    0x5d7f_bcce_d99a_61ab_502e_1b9d_b857_74f1, 0x5e73_d928_8747_8543_b588_ad1f_b491_77db,
    0x5f69_67b7_236a_b76d_d274_bd51_8da4_2e06, 0x6060_691a_6a5e_188d_e89c_c4d2_8164_f18a,
    0x6158_ddf1_b188_887a_77f8_50ac_9672_2777, 0x6252_c6db_e80d_d98f_4f9c_d8da_6b91_d86b,
    0x634e_2477_977d_1bbe_9eec_40b3_b27b_f6b2, 0x644a_f762_e47d_06e7_bb1b_7dba_1b87_aa48,
    0x6549_403b_8f76_8b97_ebf7_c10a_5655_7df6, 0x6648_ff9e_f53d_9137_022f_bf12_5c56_1132,
    0x674a_362a_0fb7_e87f_b397_beab_8c5f_6f65, 0x684c_e479_7682_7902_b96d_0bff_396f_872c,
    0x6951_0b29_5f94_b053_6c3e_7649_bef1_6b25, 0x6a56_aad5_9fe2_395d_13b3_6d7c_5da0_ddad,
    0x6b5d_c419_abfb_0241_50fa_1b26_2d3e_ead0, 0x6c66_5790_98a9_9701_ec5d_2ce9_e2ec_5570,
    0x6d70_65d5_1b8f_d71a_dcb8_23c1_eb65_69f0, 0x6e7b_ef81_8bc2_0c14_95ad_31c0_90d3_b083,
    0x6f88_f52f_e260_66fa_833d_4547_16f9_4132, 0x7097_7779_bb2e_ea86_126e_d8d6_1c2f_6152,
    0x71a7_76f8_552b_c7b3_bae8_17fd_2b86_7780, 0x72b8_f444_9324_325d_28c2_0436_3108_47bd,
    0x73cb_eff6_fc47_b359_e480_11f2_f34d_4b79, 0x74e0_6aa7_bcb9_fd8e_a53b_3743_46d5_8d4e,
    0x75f6_64ee_a623_4b3a_d306_0e06_986b_4f45, 0x770d_df63_303f_48bb_a1ca_711a_c1a9_976a,
    0x7826_da9c_796a_91e4_93eb_6d15_5287_e6ef, 0x7941_5731_472e_c6f6_1c75_c17c_d75b_16a8,
    0x7a5d_55b8_06cd_3e24_7f5b_b003_24db_2727, 0x7b7a_d6c6_cdc8_568a_eff4_ade6_0748_fcd0,
    0x7c99_daf3_5a6b_7151_47ac_3af7_48a6_533c, 0x7dba_62d3_1451_95b5_7d78_b3f1_1620_86a3,
    0x7edc_6efb_0cea_c495_4775_41ed_60de_b3d2, 0x8000_0000_0000_0000_0000_0000_0000_0000,
];

/// The bits of the least f32 at or above `n` / 2^[`POINT`], for an `n` of a normal f32.
const fn f32_at_or_above(n: u128) -> u32 {
    let top = POINT - n.leading_zeros();
    let shift = top - 23;
    let m = (n >> shift) as u32 + (n & ((1 << shift) - 1) != 0) as u32;
    // A mantissa that rounds up to 2^24 carries into the exponent.
    (top << 23) + m - (1 << 23)
}

/// The bits of the f32 nearest a value from `n` up to `n + 1` over 2^[`POINT`] that is not halfway
/// between two f32s, for an `n` of a normal f32.
const fn nearest_f32(n: u128) -> u32 {
    let top = POINT - n.leading_zeros();
    let shift = top - 23;
    let m = (n >> shift) as u32 + ((n >> (shift - 1)) & 1) as u32;
    (top << 23) + m - (1 << 23)
}

/// Per code from 1, the bits of the least f32 that encodes to it or above.
const THRESHOLDS: [u32; 255] = {
    let mut t = [0; 255];
    let mut c = 0;
    while c < 255 {
        t[c] = f32_at_or_above(HALF_CODES[c] + 1);
        c += 1;
    }
    t
};

/// Per code, [`srgb8_decode`].
static DECODED: [f32; 256] = {
    let mut d = [0.0; 256];
    let mut b = 1;
    while b < 256 {
        d[b] = f32::from_bits(nearest_f32(LINEAR[b]));
        b += 1;
    }
    d
};

/// The bits of 1.0.
const ONE: u32 = 0x3F80_0000;

/// The bits of 2^-13, where [`BUCKETS`] start, below the least threshold.
const LOW: u32 = 0x3900_0000;

/// The low bits of f32s that share a bucket, which leave 7 bits of the mantissa to index them.
const SPAN: u32 = 16;

/// The f32s from [`LOW`] to 1 in buckets of 2^[`SPAN`], each holding at most one threshold. An
/// entry holds the code where its bucket starts above `SPAN + 1` bits of the threshold's offset in
/// the bucket, which are 2^`SPAN` in a bucket with none.
static BUCKETS: [u32; ((ONE - LOW) >> SPAN) as usize + 1] = {
    const N: usize = ((ONE - LOW) >> SPAN) as usize + 1;
    assert!(THRESHOLDS[0] > LOW);
    let mut out = [0; N];
    let mut c = 0;
    let mut i = 0;
    while i < N {
        let start = LOW + ((i as u32) << SPAN);
        while c < 255 && THRESHOLDS[c] <= start {
            c += 1;
        }
        let offset = if c < 255 && THRESHOLDS[c] - start < 1 << SPAN {
            assert!(c == 254 || THRESHOLDS[c + 1] - start >= 1 << SPAN);
            THRESHOLDS[c] - start
        } else {
            1 << SPAN
        };
        out[i] = ((c as u32) << (SPAN + 1)) | offset;
        i += 1;
    }
    out
};

/// The 8-bit code of linear `linear` clamped to 0..=1: 255 times its sRGB encoding, rounded to
/// the nearest integer, which is never a tie. NaN gives 0.
#[inline]
pub fn srgb8_encode(linear: f32) -> u8 {
    // Positive f32s order as their bits.
    let bits = if linear > 0.0 {
        linear.to_bits().min(ONE)
    } else {
        0
    };
    let d = bits.max(LOW) - LOW;
    let e = BUCKETS[(d >> SPAN) as usize];
    let past = d & ((1 << SPAN) - 1) >= e & ((1 << (SPAN + 1)) - 1);
    (e >> (SPAN + 1)) as u8 + u8::from(past)
}

/// The f32 nearest the linear value of 8-bit sRGB code `code`.
#[inline]
pub fn srgb8_decode(code: u8) -> f32 {
    DECODED[usize::from(code)]
}

/// The integer nearest 2^`bits` times the linear value of 8-bit sRGB code `code`, for `bits` below
/// 127.
pub fn srgb8_decode_fixed(code: u8, bits: u32) -> u128 {
    let shift = POINT - bits;
    (LINEAR[usize::from(code)] + (1 << (shift - 1))) >> shift
}

/// [`srgb8_encode`] of `value` / 2^`bits`, exactly, for `bits` up to 127.
pub fn srgb8_encode_fixed(value: u128, bits: u32) -> u8 {
    if value >> bits != 0 {
        return 255;
    }
    let x = value << (POINT - bits);
    HALF_CODES.partition_point(|&t| t < x) as u8
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering::{self, Greater, Less};

    use num_bigint::BigUint;

    use super::*;

    fn big(v: u128) -> BigUint {
        BigUint::from(v)
    }

    /// `x` as `n / 2^POINT`.
    fn grid(x: f32) -> BigUint {
        let (m, e) = (x.to_bits() & 0x7F_FFFF | 1 << 23, x.to_bits() >> 23);
        big(m.into()) << (e + POINT - 127 - 23)
    }

    /// 255 times the linear segment of the encoding, 12.92 = 323/25 times `n / d`, against
    /// `c - 1/2`.
    fn linear_vs(n: &BigUint, d: &BigUint, c: u32) -> Ordering {
        (n * big(2 * 255 * 323)).cmp(&(d * big(25 * (2 * u128::from(c) - 1))))
    }

    /// 255 times the power segment of the encoding, `1.055 x^(5/12) - 0.055` of `x = n / d`,
    /// against `c - 1/2`: `x^(5/12)` against `(40c + 541) / 10761`, raised to the 12th power.
    fn power_vs(n: &BigUint, d: &BigUint, c: u32) -> Ordering {
        let r = big(40 * u128::from(c) + 541);
        (n.pow(5) * big(10761).pow(12)).cmp(&(r.pow(12) * d.pow(5)))
    }

    /// 255 times the encoding of `n / d`, linear to 0.0031308, against `c - 1/2`.
    fn vs_half_code(n: &BigUint, d: &BigUint, c: u32) -> Ordering {
        if n * big(10_000_000) <= d * big(31308) {
            linear_vs(n, d, c)
        } else {
            power_vs(n, d, c)
        }
    }

    /// `n / 2^POINT` against the decoding of `b / 255`: `b / 255 / 12.92` to code 10, the last
    /// at or below 0.04045, and `((b / 255 + 0.055) / 1.055)^2.4 = ((40b + 561) / 10761)^(12/5)`
    /// above.
    fn vs_linear(n: &BigUint, b: u32) -> Ordering {
        let one = big(1) << POINT;
        if b <= 10 {
            (n * big(255 * 323)).cmp(&(big(25 * u128::from(b)) * one))
        } else {
            let r = big(40 * u128::from(b) + 561);
            (n.pow(5) * big(10761).pow(12)).cmp(&(r.pow(12) * one.pow(5)))
        }
    }

    /// The greatest `n` to 2^POINT where `below(n)`, which holds from 0 to some point and not
    /// past it.
    fn last(below: impl Fn(&BigUint) -> bool) -> u128 {
        let (mut lo, mut hi) = (0, 1 << POINT);
        if below(&big(hi)) {
            return hi;
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if below(&big(mid)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    #[test]
    fn tables_are_the_exact_definition() {
        // Each segment rises, and the step between them stays inside code 10, so the code
        // rises with the value and a search finds where it passes each half code.
        let (split, d) = (big(31308), big(10_000_000));
        assert_eq!(linear_vs(&split, &d, 11), Less);
        assert_eq!(power_vs(&split, &d, 10), Greater);
        let one = big(1) << POINT;
        for c in 1..=255 {
            let t = last(|n| vs_half_code(n, &one, c) == Less);
            assert_eq!(vs_half_code(&big(t + 1), &one, c), Greater, "code {c}");
            assert_eq!(HALF_CODES[c as usize - 1], t, "code {c}");
        }
        for b in 0..=255 {
            let l = last(|n| vs_linear(n, b) != Greater);
            assert_eq!(LINEAR[b as usize], l, "code {b}");
        }
    }

    #[test]
    fn codes_change_at_the_exact_thresholds() {
        let one = big(1) << POINT;
        for c in 1..=255u8 {
            let t = f32::from_bits(THRESHOLDS[usize::from(c) - 1]);
            let below = t.next_down();
            assert_eq!(vs_half_code(&grid(t), &one, c.into()), Greater, "code {c}");
            assert_eq!(vs_half_code(&grid(below), &one, c.into()), Less, "code {c}");
            assert_eq!((srgb8_encode(below), srgb8_encode(t)), (c - 1, c));
        }
        // Each bucket steps once at most, so with the steps above, its ends pin it.
        let code = |bits: u32| THRESHOLDS.partition_point(|&t| t <= bits) as u8;
        for start in (LOW..=ONE).step_by(1 << SPAN) {
            for bits in [start, (start + (1 << SPAN) - 1).min(ONE)] {
                assert_eq!(srgb8_encode(f32::from_bits(bits)), code(bits), "{bits:#x}");
            }
        }
        assert_eq!(srgb8_encode(f32::from_bits(LOW).next_down()), 0);
    }

    #[test]
    fn codes_decode_to_the_nearest_f32_and_back() {
        assert_eq!(srgb8_decode(0).to_bits(), 0);
        for b in 1..=255u8 {
            let f = srgb8_decode(b);
            let (m, e) = (grid(f), f.to_bits() >> 23);
            let half_step = big(1) << (e + POINT - 127 - 24);
            assert_eq!(vs_linear(&(&m - &half_step), b.into()), Less, "code {b}");
            assert_eq!(vs_linear(&(&m + &half_step), b.into()), Greater, "code {b}");
            let v = big(srgb8_decode_fixed(b, 64)) << (POINT - 64);
            let half = big(1) << (POINT - 65);
            assert_eq!(vs_linear(&(&v - &half), b.into()), Less, "code {b}");
            assert_eq!(vs_linear(&(&v + &half), b.into()), Greater, "code {b}");
            assert_eq!(srgb8_encode(f), b);
            assert_eq!(srgb8_encode_fixed(srgb8_decode_fixed(b, 64), 64), b);
        }
    }

    #[test]
    fn encode_clamps_to_0_and_1() {
        let nan = f32::NAN;
        for v in [
            nan,
            -nan,
            f32::from_bits(0x7FC0_1234),
            -0.0,
            0.0,
            -1.0,
            f32::NEG_INFINITY,
        ] {
            assert_eq!(srgb8_encode(v), 0, "{v:?}");
        }
        assert_eq!(srgb8_encode(f32::from_bits(1)), 0);
        for v in [1.0, 1.0f32.next_up(), 2.0, f32::MAX, f32::INFINITY] {
            assert_eq!(srgb8_encode(v), 255, "{v:?}");
        }
        assert_eq!(srgb8_encode_fixed(1 << 64, 64), 255);
        assert_eq!(srgb8_encode_fixed(0, 64), 0);
    }
}

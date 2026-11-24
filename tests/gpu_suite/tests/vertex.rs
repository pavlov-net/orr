use gpu_suite::shaders::vertex;
use gpu_suite::{assert_clean, device, readback, render, run, upload};
use orr_gpu::{
    ClearColor, ColorAttachment, ColorLoad, ColorTarget, Format, Hazards, MemoryKind,
    PrimitiveTopology, Rendering, ShaderStage, SpecData, Stages, TextureDesc, TextureMemory,
    TextureRegion, TextureUsage, VertexPipelineDesc,
};

#[test]
fn draws_lists_and_strips_with_vertex_and_instance_ids() {
    let gpu = device();
    let stage = |name| ShaderStage::new(&vertex::MODULE, name, SpecData::default()).unwrap();
    let colors = [ColorTarget::new(Format::R32Uint)];
    let pipeline = |topology| {
        gpu.create_vertex_pipeline(
            &VertexPipelineDesc {
                topology,
                pixel: Some(stage("pixel_main")),
                colors: &colors,
                ..VertexPipelineDesc::new(stage("vertex_main"))
            },
            "vertex test",
        )
        .unwrap()
    };
    let strips = pipeline(PrimitiveTopology::TriangleStrip);
    let triangles = pipeline(PrimitiveTopology::TriangleList);

    let rectangles: [[f32; 4]; 4] = [
        [-1.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ];
    let data = upload(&rectangles, "rectangle upload");
    let vertex_data = gpu
        .gpu_malloc::<[f32; 4]>(MemoryKind::DeviceLocal, 4, "rectangles")
        .unwrap();
    let root = |first, id, list| {
        upload(
            &[vertex::Root {
                rectangles: vertex_data.gpu().offset(first),
                first_id: id,
                triangle_list: list,
            }],
            "vertex root",
        )
    };
    let top = root(0, 11, 0);
    let bottom = root(2, 21, 1);
    let desc = TextureDesc::d2(Format::R32Uint, 8, 4, TextureUsage::COLOR);
    let target = gpu
        .create_texture(&desc, TextureMemory::Dedicated, "vertex test")
        .unwrap();
    let out = readback::<u32>(32, "vertex pixels");

    let mut list = render().command_list().unwrap();
    list.copy(
        vertex_data.gpu().cast(),
        data.gpu().cast(),
        size_of_val(&rectangles) as u64,
    );
    list.barrier(Stages::TRANSFER, Stages::VERTEX, Hazards::empty());
    list.discard(&target);
    list.begin_rendering(&Rendering {
        extent: [8, 4],
        colors: &[ColorAttachment {
            texture: &target,
            load: ColorLoad::Clear(ClearColor::Uint([0; 4])),
            store: true,
        }],
        depth: None,
    });
    list.draw(&strips, top.gpu(), 4, 2);
    list.draw(&triangles, bottom.gpu(), 6, 2);
    list.end_rendering();
    list.barrier(Stages::COLOR_OUT, Stages::TRANSFER, Hazards::empty());
    list.copy_from_texture(
        out.gpu().cast(),
        &target,
        &TextureRegion::whole_mip(&desc, 0),
    );
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    run(list);

    assert_eq!(
        out.as_slice().unwrap(),
        [
            11, 11, 11, 11, 12, 12, 12, 12, 11, 11, 11, 11, 12, 12, 12, 12, 21, 0, 0, 21, 22, 0, 0,
            22, 0, 0, 0, 0, 0, 0, 0, 0,
        ]
    );
    assert_clean();
}

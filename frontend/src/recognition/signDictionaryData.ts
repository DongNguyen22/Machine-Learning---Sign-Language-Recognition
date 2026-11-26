export type SignGuideEntry = {
  label: string;
  category: "Chữ số (0–9)" | "Chữ cái (A–Z)";
  imageUrl: string | null;
  sourceFilename: string;
  shortSummary: string;
  handPosture: string;
  palmAndCameraView: string;
  motionAndDatasetNote: string;
  hasMotionOrSpecialLimit: boolean;
};

const STANDARD_STATIC_NOTE =
  "Ảnh mẫu trích từ dữ liệu huấn luyện (asl_processed/train) dùng cho mô hình phân loại ảnh tĩnh. Trong ảnh mẫu của dataset, phần lớn ký hiệu có ngón cái/cạnh ngoài nằm ở phía bên phải khung ảnh; đây là ảnh mẫu dữ liệu huấn luyện, không phải giáo trình ASL chuẩn.";

export const SIGN_DICTIONARY_MAP: Record<string, SignGuideEntry> = {
  "0": {
    label: "0",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/0.jpg",
    sourceFilename: "P7_0_616.jpg",
    shortSummary: "Bàn tay trước uốn vòng chữ O đặt trước lòng bàn tay sau xòe thẳng.",
    handPosture:
      "Trong ảnh mẫu của dataset, ký hiệu 0 kết hợp hai bàn tay: bàn tay phía sau duỗi thẳng 4 ngón hướng lên, bàn tay phía trước uốn cong các ngón chạm đầu ngón cái tạo vòng chữ O.",
    palmAndCameraView:
      "Lòng bàn tay sau hướng về phía camera; bàn tay chữ O phía trước đặt nghiêng ngang để camera thấy rõ vòng tròn giữa ngón cái và các ngón.",
    motionAndDatasetNote:
      "Lưu ý giới hạn dữ liệu huấn luyện: Mẫu lớp 0 trong bộ ASL-HG dùng 2 bàn tay chồng lên nhau, nhưng bộ cắt MediaPipe khi nhận diện trực tiếp chỉ lấy khung 1 bàn tay lớn nhất. Đây là ảnh mẫu trong dữ liệu huấn luyện, không phải hướng dẫn ASL chuẩn.",
    hasMotionOrSpecialLimit: true,
  },
  "1": {
    label: "1",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/1.jpg",
    sourceFilename: "P2_1_116.jpg",
    shortSummary: "Giơ thẳng duy nhất ngón trỏ hướng lên; các ngón khác nắm gọn.",
    handPosture:
      "Duỗi thẳng ngón trỏ hướng thẳng đứng lên trên; gập ba ngón (giữa, áp út, út) vào lòng bàn tay và dùng ngón cái đè giữ phía trước.",
    palmAndCameraView:
      "Lòng bàn tay hướng trực diện về phía camera; trên ảnh mẫu, ngón trỏ nằm ở phía bên phải của nắm tay.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  "2": {
    label: "2",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/2.jpg",
    sourceFilename: "P10_2_914.jpg",
    shortSummary: "Giơ hai ngón trỏ và giữa tách nhẹ hình chữ V; ngón cái giữ hai ngón còn lại.",
    handPosture:
      "Duỗi thẳng ngón trỏ và ngón giữa hướng lên trên, tách nhẹ thành góc chữ V; gập ngón áp út và ngón út vào lòng bàn tay, ngón cái đè ngang lên trên hai ngón gập.",
    palmAndCameraView:
      "Lòng bàn tay hướng thẳng về phía camera, giữ cổ tay thẳng đứng để hai ngón giơ lên nằm gọn giữa khung hình.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  "3": {
    label: "3",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/3.jpg",
    sourceFilename: "P10_3_921.jpg",
    shortSummary: "Xòe ba ngón gồm ngón cái, ngón trỏ và ngón giữa; gập ngón áp út và ngón út.",
    handPosture:
      "Duỗi thẳng ngón trỏ và ngón giữa hướng lên, đồng thời mở rộng ngón cái sang bên cạnh; gập hai ngón áp út và ngón út sát vào lòng bàn tay.",
    palmAndCameraView:
      "Lòng bàn tay hướng trực diện về phía camera; trên ảnh mẫu, ngón cái xòe ra ở phía bên phải khung ảnh.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  "4": {
    label: "4",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/4.jpg",
    sourceFilename: "P10_4_921.jpg",
    shortSummary: "Xòe thẳng 4 ngón (trỏ, giữa, áp út, út) tách rời; gập ngón cái vào lòng bàn tay.",
    handPosture:
      "Duỗi thẳng và tách đều bốn ngón (trỏ, giữa, áp út, út) hướng lên trên; gập ngón cái ép gọn ngang qua lòng bàn tay.",
    palmAndCameraView:
      "Lòng bàn tay hướng thẳng về phía camera sao cho thấy rõ 4 ngón tách rời và phần ngón cái gập phía trước.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  "5": {
    label: "5",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/5.jpg",
    sourceFilename: "P10_5_913.jpg",
    shortSummary: "Xòe rộng cả 5 ngón tay tách rời nhau, lòng bàn tay mở phẳng.",
    handPosture:
      "Duỗi thẳng và xòe rộng cả 5 ngón tay (ngón cái, trỏ, giữa, áp út, út) với khoảng cách đều nhau.",
    palmAndCameraView:
      "Lòng bàn tay mở phẳng hướng trực diện về phía camera; trên ảnh mẫu, ngón cái mở rộng sang phía bên phải khung ảnh.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  "6": {
    label: "6",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/6.jpg",
    sourceFilename: "P2_6_104.jpg",
    shortSummary: "Giơ 3 ngón (trỏ, giữa, áp út); chụm đầu ngón út chạm đầu ngón cái.",
    handPosture:
      "Duỗi thẳng ba ngón (trỏ, giữa, áp út) tách nhẹ hướng lên trên; đưa ngón út và ngón cái vòng vào chạm đầu nhau ở phía trước lòng bàn tay.",
    palmAndCameraView:
      "Lòng bàn tay hướng thẳng về phía camera để thấy rõ vòng chạm giữa ngón cái và ngón út ở phía dưới.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  "7": {
    label: "7",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/7.jpg",
    sourceFilename: "P10_7_918.jpg",
    shortSummary: "Giơ 3 ngón (trỏ, giữa, út); gập ngón áp út xuống chạm đầu ngón cái.",
    handPosture:
      "Duỗi thẳng ngón trỏ, ngón giữa và ngón út; gập riêng ngón áp út xuống để đầu ngón áp út chạm vào đầu ngón cái trước lòng bàn tay.",
    palmAndCameraView:
      "Lòng bàn tay hướng về phía camera; trên ảnh mẫu, ngón út xòe sang bên trái còn hai ngón trỏ và giữa hướng lên bên phải.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  "8": {
    label: "8",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/8.jpg",
    sourceFilename: "P10_8_921.jpg",
    shortSummary: "Giơ 3 ngón (trỏ, áp út, út); gập ngón giữa xuống chạm đầu ngón cái.",
    handPosture:
      "Duỗi thẳng ngón trỏ, ngón áp út và ngón út; gập riêng ngón giữa xuống chạm vào đầu ngón cái ở chính giữa lòng bàn tay.",
    palmAndCameraView:
      "Lòng bàn tay hướng trực diện về phía camera để phân biệt rõ ngón giữa đang gập chạm ngón cái.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  "9": {
    label: "9",
    category: "Chữ số (0–9)",
    imageUrl: "/signs/9.jpg",
    sourceFilename: "P10_9_905.jpg",
    shortSummary: "Duỗi 3 ngón (giữa, áp út, út); uốn ngón trỏ chạm đầu ngón cái.",
    handPosture:
      "Duỗi thẳng và tách ba ngón (giữa, áp út, út) hướng lên trên; uốn cong ngón trỏ xuống chạm đầu ngón cái tạo vòng tròn nhỏ.",
    palmAndCameraView:
      "Lòng bàn tay hướng về phía camera; trên ảnh mẫu, vòng chạm giữa ngón trỏ và ngón cái nằm ở phía bên phải khung ảnh.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  A: {
    label: "A",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/A.jpg",
    sourceFilename: "P10_A_907.jpg",
    shortSummary: "Nắm gọn 4 ngón tay; đặt ngón cái áp thẳng dọc theo cạnh bên của nắm tay.",
    handPosture:
      "Gập kín bốn ngón (trỏ, giữa, áp út, út) xuống lòng bàn tay tạo thành nắm đấm gọn; đặt ngón cái duỗi thẳng áp sát vào cạnh ngoài của ngón trỏ (không vắt ngang qua mặt trước các ngón).",
    palmAndCameraView:
      "Mặt trước nắm tay hướng trực diện về camera. Lưu ý quan trọng từ ảnh mẫu huấn luyện: ngón cái nằm ở mép bên phải của nắm tay trên ảnh.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  B: {
    label: "B",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/B.jpg",
    sourceFilename: "P1_B_100.jpg",
    shortSummary: "Duỗi thẳng và khép sát 4 ngón tay hướng lên; gập ngón cái ngang lòng bàn tay.",
    handPosture:
      "Duỗi thẳng đứng cả bốn ngón (trỏ, giữa, áp út, út) và ép sát các ngón vào nhau tạo thành mặt phẳng; gập ngón cái vắt ngang sát vào lòng bàn tay.",
    palmAndCameraView:
      "Lòng bàn tay hướng thẳng, vuông góc với ống kính camera; khung bàn tay có dáng chữ nhật đứng thon gọn.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  C: {
    label: "C",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/C.jpg",
    sourceFilename: "P2_C_107.jpg",
    shortSummary: "Khép các ngón và uốn cong bàn tay cùng ngón cái tạo thành vòm chữ C.",
    handPosture:
      "Khép bốn ngón trên và uốn cong đều cùng với ngón cái ở phía dưới sao cho đường viền bàn tay tạo thành nửa vòng tròn mở hình chữ C.",
    palmAndCameraView:
      "Xoay nghiêng bàn tay để ống kính camera nhìn thẳng vào mặt bên, thấy rõ khoảng mở chữ C hướng sang bên phải trên ảnh mẫu.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  D: {
    label: "D",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/D.jpg",
    sourceFilename: "P10_D_923.jpg",
    shortSummary: "Giơ thẳng ngón trỏ lên trên; ba ngón còn lại uốn tròn chạm đầu ngón cái.",
    handPosture:
      "Duỗi thẳng duy nhất ngón trỏ hướng thẳng đứng lên trên; uốn cong ba ngón (giữa, áp út, út) xuống chạm vào đầu ngón cái tạo thành cụm tròn phía dưới.",
    palmAndCameraView:
      "Lòng bàn tay hướng về phía camera (hơi xoay nhẹ) sao cho ngón trỏ nổi bật thẳng đứng ở phía bên phải khung ảnh mẫu.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  E: {
    label: "E",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/E.jpg",
    sourceFilename: "P10_E_913.jpg",
    shortSummary: "Gập cong các đốt ngón tay xuống; ngón cái gập ngang đỡ dưới đầu các ngón.",
    handPosture:
      "Co gập cả bốn ngón (trỏ, giữa, áp út, út) xuống sao cho các đầu ngón tay tựa sát lên phần ngón cái đang gập nằm ngang ở phía dưới.",
    palmAndCameraView:
      "Mặt trước bàn tay hướng trực diện về phía camera, giữ các ngón khép kín tạo khối tròn gọn.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  F: {
    label: "F",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/F.jpg",
    sourceFilename: "P10_F_904.jpg",
    shortSummary: "Chụm ngón trỏ và ngón cái thành vòng tròn; khép và duỗi thẳng 3 ngón còn lại.",
    handPosture:
      "Uốn ngón trỏ xuống chạm đầu ngón cái tạo vòng tròn nhỏ; ba ngón còn lại (giữa, áp út, út) duỗi thẳng đứng lên trên.",
    palmAndCameraView:
      "Lòng bàn tay hướng về phía camera; trong ảnh mẫu, ba ngón duỗi thẳng được khép sát nhau ở phía bên trái còn vòng tròn ngón trỏ–cái nằm ở bên phải.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  G: {
    label: "G",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/G.jpg",
    sourceFilename: "P2_G_105.jpg",
    shortSummary: "Đặt bàn tay ngang; duỗi ngón trỏ chỉ sang ngang, ngón cái áp song song.",
    handPosture:
      "Gập ba ngón (giữa, áp út, út) vào lòng bàn tay; duỗi thẳng ngón trỏ hướng ngang sang bên trái ảnh, ngón cái đặt dọc phía trên/trước ngón giữa.",
    palmAndCameraView:
      "Lòng bàn tay hướng vào người thực hiện hoặc hơi nghiêng về camera, tạo khung hình chữ nhật nằm ngang.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  H: {
    label: "H",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/H.jpg",
    sourceFilename: "P2_H_106.jpg",
    shortSummary: "Đặt bàn tay ngang; duỗi thẳng và khép sát hai ngón trỏ và giữa chỉ sang ngang.",
    handPosture:
      "Duỗi thẳng đồng thời ngón trỏ và ngón giữa khép sát song song chỉ ngang sang bên trái ảnh; gập ngón áp út và ngón út vào trong với ngón cái đặt giữ phía trước.",
    palmAndCameraView:
      "Bàn tay đặt nằm ngang trước ống kính camera, thấy rõ hai ngón duỗi ngang dài về bên trái khung hình.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  I: {
    label: "I",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/I.jpg",
    sourceFilename: "P10_I_917.jpg",
    shortSummary: "Nắm gọn 3 ngón và ngón cái; chỉ giơ thẳng duy nhất ngón út hướng lên.",
    handPosture:
      "Gập ba ngón (trỏ, giữa, áp út) vào lòng bàn tay và đặt ngón cái giữ phía trước; duỗi thẳng ngón út hướng lên trên.",
    palmAndCameraView:
      "Mặt lòng nắm tay hướng về phía camera; trên ảnh mẫu, ngón út nhô lên ở góc trên bên trái của nắm tay.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  J: {
    label: "J",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/J.jpg",
    sourceFilename: "P2_J_113.jpg",
    shortSummary: "Giơ ngón út như chữ I nhưng nghiêng chéo bàn tay sang trái trong khung hình tĩnh.",
    handPosture:
      "Giữ nắm tay với ngón út duỗi thẳng tương tự chữ I, nhưng xoay nghiêng cổ tay để ngón út chĩa chéo sang góc trên bên trái khung ảnh (khung hình rộng ngang hơn chữ I).",
    palmAndCameraView:
      "Lòng bàn tay hướng về camera với cổ tay nghiêng chéo sang trái.",
    motionAndDatasetNote:
      "Lưu ý giới hạn chuyển động: Trong ASL chuẩn, ký hiệu J yêu cầu dùng ngón út vẽ đường móc chữ J trong không trung. Tuy nhiên, dữ liệu huấn luyện và mô hình phân loại ảnh tĩnh chỉ xử lý 1 khung hình tại tư thế nghiêng chéo như ảnh mẫu này (không phải hướng dẫn ASL chuyển động chuẩn).",
    hasMotionOrSpecialLimit: true,
  },
  K: {
    label: "K",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/K.jpg",
    sourceFilename: "P2_K_117.jpg",
    shortSummary: "Giơ ngón trỏ và ngón giữa tách nhẹ; đặt đầu ngón cái vào giữa gốc hai ngón.",
    handPosture:
      "Duỗi thẳng ngón trỏ và ngón giữa hướng lên trên (tách nhẹ chữ V); đưa ngón cái hướng lên chạm vào khe giữa đốt gốc của ngón trỏ và ngón giữa; gập hai ngón còn lại.",
    palmAndCameraView:
      "Lòng bàn tay hướng về phía camera sao cho thấy rõ ngón cái đặt dọc ở giữa gốc hai ngón giơ lên.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  L: {
    label: "L",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/L.jpg",
    sourceFilename: "P10_L_901.jpg",
    shortSummary: "Duỗi thẳng ngón trỏ lên trên và mở ngón cái vuông góc sang ngang tạo chữ L.",
    handPosture:
      "Duỗi thẳng đứng ngón trỏ hướng lên trên và mở thẳng ngón cái vuông góc sang bên cạnh (góc 90 độ); gập gọn ba ngón (giữa, áp út, út) vào lòng bàn tay.",
    palmAndCameraView:
      "Lòng bàn tay hướng trực diện về phía camera; trên ảnh mẫu, ngón cái mở sang phía bên phải tạo đúng hình chữ L xuôi.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  M: {
    label: "M",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/M.jpg",
    sourceFilename: "P10_M_918.jpg",
    shortSummary: "Luồn ngón cái vào trong lòng bàn tay dưới 3 ngón (trỏ, giữa, áp út) gập phủ lên.",
    handPosture:
      "Đưa ngón cái vắt ngang sâu vào lòng bàn tay tới sát ngón út, sau đó gập ba ngón (trỏ, giữa, áp út) phủ đè lên trên ngón cái.",
    palmAndCameraView:
      "Mặt trước nắm tay hướng trực diện về phía camera, giữ nắm tay gọn gần như hình vuông.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  N: {
    label: "N",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/N.jpg",
    sourceFilename: "P10_N_904.jpg",
    shortSummary: "Luồn ngón cái dưới 2 ngón (trỏ và giữa) sao cho đầu ngón cái kẹp trước ngón áp út.",
    handPosture:
      "Đưa ngón cái luồn qua dưới ngón trỏ và ngón giữa, rồi gập các ngón tay lại sao cho hai ngón trỏ và giữa phủ lên trên ngón cái.",
    palmAndCameraView:
      "Mặt trước nắm tay hướng thẳng về phía camera.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  O: {
    label: "O",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/O.jpg",
    sourceFilename: "P2_O_110.jpg",
    shortSummary: "Uốn cong cả 4 ngón trên chạm vào đầu ngón cái tạo thành vòng tròn chữ O khép kín.",
    handPosture:
      "Khép các ngón tay và uốn cong đều xuống sao cho đầu các ngón chạm khít vào đầu ngón cái, tạo thành một lỗ tròn ở giữa giống chữ O.",
    palmAndCameraView:
      "Xoay nghiêng bàn tay (tương tự góc nhìn chữ C) để ống kính camera nhìn xuyên qua lỗ tròn chữ O.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  P: {
    label: "P",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/P.jpg",
    sourceFilename: "P2_P_111.jpg",
    shortSummary: "Bàn tay nằm ngang; ngón trỏ chỉ thẳng sang ngang, ngón giữa và ngón cái chúc xuống.",
    handPosture:
      "Duỗi thẳng ngón trỏ chỉ ngang sang bên trái ảnh; hạ ngón giữa chúc xuống dưới tạo góc mở với ngón trỏ, đầu ngón cái chạm vào ngón giữa.",
    palmAndCameraView:
      "Bàn tay đặt ngang nhìn từ mặt bên để camera thấy rõ hình gọng kìm mở giữa ngón trỏ (trên) và ngón giữa/cái (dưới).",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  Q: {
    label: "Q",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/Q.jpg",
    sourceFilename: "P2_Q_116.jpg",
    shortSummary: "Cổ tay gập chúc xuống; duỗi ngón trỏ và ngón cái cùng hướng chếch xuống dưới.",
    handPosture:
      "Gập cổ tay hướng xuống; duỗi ngón trỏ và ngón cái mở song song/hơi tách nhau chĩa chếch xuống phía dưới bên trái ảnh; ba ngón còn lại co gọn.",
    palmAndCameraView:
      "Camera nhìn từ mặt bên của bàn tay đang chúc xuống dưới.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  R: {
    label: "R",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/R.jpg",
    sourceFilename: "P10_R_920.jpg",
    shortSummary: "Giơ thẳng ngón trỏ và ngón giữa hướng lên và bắt chéo hai ngón vào nhau.",
    handPosture:
      "Duỗi thẳng ngón trỏ và ngón giữa hướng lên trên rồi vắt chéo ngón giữa qua ngón trỏ; gập ngón áp út và ngón út xuống dưới ngón cái.",
    palmAndCameraView:
      "Lòng bàn tay hướng thẳng về phía camera, giữ khung bàn tay thẳng đứng.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  S: {
    label: "S",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/S.jpg",
    sourceFilename: "P10_S_906.jpg",
    shortSummary: "Nắm chặt 4 ngón tay; đặt ngón cái vắt ngang đè lên mặt trước các ngón đang nắm.",
    handPosture:
      "Gập kín bốn ngón (trỏ, giữa, áp út, út) vào lòng bàn tay thành nắm đấm; gập ngón cái vắt ngang qua mặt trước của ngón trỏ và ngón giữa (khác với chữ A có ngón cái đặt dọc bên cạnh).",
    palmAndCameraView:
      "Mặt trước nắm tay hướng trực diện về phía camera để thấy rõ ngón cái vắt ngang phía trước.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  T: {
    label: "T",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/T.jpg",
    sourceFilename: "P10_T_913.jpg",
    shortSummary: "Nắm các ngón tay lại và kẹp đầu ngón cái nhô lên giữa ngón trỏ và ngón giữa.",
    handPosture:
      "Đặt ngón cái luồn vào khe giữa ngón trỏ và ngón giữa sao cho đầu ngón cái nhô lên phía trên, các ngón gập chặt xung quanh.",
    palmAndCameraView:
      "Mặt trước nắm tay hướng thẳng về phía camera.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  U: {
    label: "U",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/U.jpg",
    sourceFilename: "P1_U_19.jpg",
    shortSummary: "Duỗi thẳng và khép sát hai ngón trỏ và giữa song song hướng lên trên.",
    handPosture:
      "Duỗi thẳng đồng thời ngón trỏ và ngón giữa và ép sát khít vào nhau (không tách chữ V); gập ngón áp út và ngón út xuống có ngón cái giữ phía trước.",
    palmAndCameraView:
      "Lòng bàn tay hướng về phía camera, hai ngón khép sát tạo dáng thon cao.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  V: {
    label: "V",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/V.jpg",
    sourceFilename: "P10_V_927.jpg",
    shortSummary: "Duỗi thẳng hai ngón trỏ và giữa tách rộng thành hình chữ V; các ngón khác nắm gọn.",
    handPosture:
      "Duỗi thẳng ngón trỏ và ngón giữa hướng lên trên và tách rộng hai ngón tạo thành hình chữ V rõ rệt; gập hai ngón áp út và út xuống dưới ngón cái.",
    palmAndCameraView:
      "Lòng bàn tay hướng trực diện về phía camera.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  W: {
    label: "W",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/W.jpg",
    sourceFilename: "P10_W_927.jpg",
    shortSummary: "Duỗi thẳng và tách đều 3 ngón (trỏ, giữa, áp út) tạo hình chữ W; ngón cái giữ ngón út.",
    handPosture:
      "Duỗi thẳng ba ngón (trỏ, giữa, áp út) hướng lên và tách đều nhau tạo hình chữ W; gập ngón út vào lòng bàn tay và dùng ngón cái chạm giữ ngón út.",
    palmAndCameraView:
      "Lòng bàn tay hướng thẳng về phía camera.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  X: {
    label: "X",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/X.jpg",
    sourceFilename: "P2_X_121.jpg",
    shortSummary: "Nắm 3 ngón dưới và ngón cái; giơ ngón trỏ lên và uốn cong thành hình móc câu.",
    handPosture:
      "Gập gọn ba ngón (giữa, áp út, út) vào lòng bàn tay có ngón cái đè giữ phía trước; đưa riêng ngón trỏ lên cao và gập cong hai đốt trên thành hình chiếc móc.",
    palmAndCameraView:
      "Lòng bàn tay hướng về camera (hơi nghiêng nhẹ) để thấy rõ độ cong hình móc của ngón trỏ ở phía bên phải ảnh mẫu.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  Y: {
    label: "Y",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/Y.jpg",
    sourceFilename: "P10_Y_911.jpg",
    shortSummary: "Xòe rộng ngón cái và ngón út sang hai bên; gập gọn 3 ngón ở giữa vào lòng bàn tay.",
    handPosture:
      "Duỗi thẳng ngón cái mở sang phải và ngón út mở sang trái; gập kín ba ngón ở giữa (trỏ, giữa, áp út) xuống lòng bàn tay.",
    palmAndCameraView:
      "Lòng bàn tay hướng trực diện về phía camera, tạo dáng xòe ngang cân đối hai đầu.",
    motionAndDatasetNote: STANDARD_STATIC_NOTE,
    hasMotionOrSpecialLimit: false,
  },
  Z: {
    label: "Z",
    category: "Chữ cái (A–Z)",
    imageUrl: "/signs/Z.jpg",
    sourceFilename: "P10_Z_918.jpg",
    shortSummary: "Giơ riêng ngón trỏ chỉ chéo lên phía bên trái khung ảnh; các ngón khác nắm gọn.",
    handPosture:
      "Gập gọn ba ngón (giữa, áp út, út) và ngón cái vào lòng bàn tay; duỗi thẳng duy nhất ngón trỏ hướng chéo lên góc trên bên trái của ảnh.",
    palmAndCameraView:
      "Bàn tay xoay nhẹ hướng ngón trỏ chéo sang trái trước ống kính camera.",
    motionAndDatasetNote:
      "Lưu ý giới hạn chuyển động: Trong ASL chuẩn, chữ Z yêu cầu dùng đầu ngón trỏ vẽ đường zíc-zắc chữ Z trong không trung. Tuy nhiên, mô hình phân loại ảnh tĩnh hiện chỉ nhận diện trên 1 khung hình với ngón trỏ chỉ chéo sang trái như ảnh mẫu dữ liệu huấn luyện này.",
    hasMotionOrSpecialLimit: true,
  },
};

export function getSignGuideEntry(label: string): SignGuideEntry {
  const key = label.trim();
  const existing = SIGN_DICTIONARY_MAP[key];
  if (existing) return existing;

  const isDigit = /^[0-9]$/.test(key);
  return {
    label: key,
    category: isDigit ? "Chữ số (0–9)" : "Chữ cái (A–Z)",
    imageUrl: null,
    sourceFilename: "Chưa có trong bộ ảnh trích xuất",
    shortSummary: "Chưa có ảnh hướng dẫn cho nhãn này trong bộ dữ liệu giao diện.",
    handPosture: "Chưa có dữ liệu mô tả tư thế bàn tay đã kiểm chứng cho nhãn này.",
    palmAndCameraView: "Chưa có thông tin góc nhìn camera.",
    motionAndDatasetNote: "Trạng thái: Chưa có ảnh hướng dẫn (không gán ảnh của lớp khác).",
    hasMotionOrSpecialLimit: false,
  };
}

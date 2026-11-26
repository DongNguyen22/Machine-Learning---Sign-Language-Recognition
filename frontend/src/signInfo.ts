// Thông tin hiển thị cho từ điển ký hiệu ASL A-Z.
// Ảnh của mỗi chữ đặt tại public/images/signs/<CHỮ>.png (vd: public/images/signs/A.png).
// Nếu chưa có ảnh, giao diện tự dùng hình vẽ minh họa `pose`.

export type SignInfo = {
  description: string;
  category: "Tĩnh" | "Có chuyển động";
  pose: "open" | "fist" | "one" | "two";
};

export const SIGN_CATEGORIES = ["Tất cả", "Tĩnh", "Có chuyển động"] as const;

export const signImageUrl = (label: string) => `/images/signs/${encodeURIComponent(label)}.png`;

const S = (description: string, pose: SignInfo["pose"]): SignInfo => ({
  description,
  category: "Tĩnh",
  pose,
});
const M = (description: string, pose: SignInfo["pose"]): SignInfo => ({
  description,
  category: "Có chuyển động",
  pose,
});

export const SIGN_INFO: Record<string, SignInfo> = {
  A: S("Nắm tay lại, ngón cái áp sát cạnh ngón trỏ, hướng lên trên.", "fist"),
  B: S("Bốn ngón duỗi thẳng và khép sát nhau, ngón cái gập ngang lòng bàn tay.", "open"),
  C: S("Các ngón cong lại cùng ngón cái tạo thành hình chữ C.", "open"),
  D: S("Ngón trỏ duỗi thẳng lên, các ngón còn lại cong chạm đầu ngón cái tạo vòng tròn.", "one"),
  E: S("Gập các ngón xuống, đầu ngón chạm lên ngón cái đang gập ngang lòng bàn tay.", "fist"),
  F: S("Ngón trỏ chạm ngón cái tạo vòng tròn, ba ngón còn lại duỗi thẳng và hơi xòe.", "open"),
  G: S("Ngón trỏ và ngón cái duỗi song song chỉ sang ngang, các ngón khác nắm lại.", "one"),
  H: S("Ngón trỏ và ngón giữa duỗi khép sát chỉ sang ngang, các ngón khác nắm lại.", "two"),
  I: S("Nắm tay lại, chỉ duỗi thẳng ngón út lên trên.", "one"),
  J: M("Làm dáng chữ I rồi dùng ngón út vẽ hình chữ J trong không khí.", "one"),
  K: S("Ngón trỏ và ngón giữa duỗi lên tách chữ V, ngón cái đặt giữa hai ngón.", "two"),
  L: S("Ngón cái và ngón trỏ duỗi ra vuông góc tạo hình chữ L.", "one"),
  M: S("Ngón cái đặt dưới ba ngón trỏ, giữa và áp út đang gập xuống.", "fist"),
  N: S("Ngón cái đặt dưới hai ngón trỏ và giữa đang gập xuống.", "fist"),
  O: S("Tất cả các ngón cong lại chạm đầu ngón cái tạo thành hình chữ O.", "fist"),
  P: S("Giống chữ K nhưng xoay bàn tay chúc xuống dưới.", "two"),
  Q: S("Giống chữ G nhưng xoay bàn tay chúc xuống dưới.", "one"),
  R: S("Ngón trỏ và ngón giữa duỗi lên và bắt chéo nhau.", "two"),
  S: S("Nắm tay lại, ngón cái vắt ngang phía trước các ngón.", "fist"),
  T: S("Nắm tay lại, ngón cái kẹp giữa ngón trỏ và ngón giữa.", "fist"),
  U: S("Ngón trỏ và ngón giữa duỗi thẳng, khép sát nhau chỉ lên trên.", "two"),
  V: S("Ngón trỏ và ngón giữa duỗi thẳng, tách rời tạo hình chữ V.", "two"),
  W: S("Ngón trỏ, giữa và áp út duỗi thẳng, tách rời nhau.", "open"),
  X: S("Ngón trỏ cong lại như móc câu, các ngón khác nắm lại.", "one"),
  Y: S("Duỗi ngón cái và ngón út ra hai bên, ba ngón giữa gập lại.", "open"),
  Z: M("Dùng ngón trỏ vẽ hình chữ Z trong không khí.", "one"),
};

export const getSignInfo = (label: string): SignInfo =>
  SIGN_INFO[label.toUpperCase()] ?? S("Ký hiệu được hỗ trợ bởi model đã cấu hình.", "open");

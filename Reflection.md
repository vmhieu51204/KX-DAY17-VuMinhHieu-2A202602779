# Phân tích kết quả lab Memory Systems for AI Agent

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ lịch sử trong cùng một thread. Khi chuyển sang thread mới, agent không có thông tin từ các phiên trước nên không thể nhắc lại tên, nghề nghiệp hay sở thích của người dùng. Advanced lưu các thông tin ổn định vào file `User.md` và đưa hồ sơ này vào ngữ cảnh của mỗi lượt trả lời. 

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Ở mỗi lượt, Advanced phải xử lý thêm nội dung hồ sơ người dùng bên cạnh lịch sử hội thoại. Khi hội thoại còn ngắn và chưa vượt ngưỡng compact, phần ngữ cảnh bổ sung này tạo ra chi phí nhưng chưa có lợi ích tiết kiệm từ việc nén lịch sử.Persistent memory giúp cải thiện recall nhưng có thể làm tăng chi phí ở hội thoại ngắn. Nếu sử dụng LLM để trích xuất fact hoặc tạo summary, hệ thống còn có thể phát sinh chi phí cho các lời gọi model bổ sung; bản offline hiện tại dùng heuristic nên không có những lời gọi này.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Baseline mang theo toàn bộ lịch sử của thread ở mỗi lượt. Khi hội thoại dài hơn, các đoạn cũ bị xử lý lặp lại nhiều lần, khiến tổng lượng prompt tokens tăng nhanh. Advanced thay phần lịch sử cũ bằng summary có giới hạn kích thước, đồng thời giữ một số message gần nhất ở dạng đầy đủ. Ngữ cảnh được đưa vào prompt gồm hồ sơ người dùng, summary và các message gần nhất. Cách này giảm lượng văn bản phải xử lý lại mà vẫn giữ các fact ổn định và ngữ cảnh gần để trả lời tiếp.

## 4. File memory tăng trưởng ra sao và rủi ro gì đi kèm?

File `User.md` tăng khi agent thêm trường thông tin mới hoặc thay giá trị cũ bằng nội dung dài hơn. Trong triển khai hiện tại, các fact được lưu theo trường như `name`, `location`, `profession` và `response_style`. Việc nhắc lại cùng một fact không tạo thêm dòng trùng lặp; cập nhật cùng một trường sẽ thay thế giá trị đã có. Vì vậy, dung lượng không tăng tuyến tính theo số lượt chat và cũng có thể giảm nếu giá trị mới ngắn hơn.

Các rủi ro đi kèm gồm:

- **Lưu sai fact:** câu hỏi, câu đùa hoặc thông tin về người khác có thể bị hiểu nhầm thành thông tin của người dùng. Một fact sai đã lưu có thể ảnh hưởng nhiều phiên sau.
- **Thông tin lỗi thời:** nơi ở, nghề nghiệp và sở thích có thể thay đổi. Nếu agent bỏ sót lời đính chính, hồ sơ sẽ tiếp tục cung cấp thông tin cũ.
- **Tăng chi phí:** hồ sơ dài hơn làm tăng dung lượng lưu trữ và lượng ngữ cảnh phải xử lý ở mỗi lượt. Nhiều người dùng cũng làm tăng tổng số file.
- **Dữ liệu cá nhân:** hồ sơ lưu thông tin lâu dài nên cần được bảo vệ và có cơ chế cho người dùng xem, sửa hoặc xóa.

Triển khai hiện tại giảm một phần rủi ro bằng cách bỏ qua một số câu hỏi, câu giả định và câu đùa, đồng thời thay thế fact khi có cập nhật rõ ràng. Tuy nhiên, quy tắc trích xuất bằng regex vẫn có thể bỏ sót hoặc hiểu sai cách diễn đạt mới. Khi mở rộng hệ thống, có thể bổ sung giới hạn kích thước hồ sơ, thời điểm cập nhật, kiểm tra độ tin cậy và cơ chế hết hạn cho thông tin dễ thay đổi.

## Cơ sở số liệu

Các số liệu trên được lấy từ lệnh `python src/benchmark.py` với cấu hình mặc định: ngưỡng compact **1.200 token ước lượng** và giữ **4 message gần nhất**. Token được ước lượng từ số ký tự, không phải số token tính phí thực tế của provider. Tổng token của benchmark bao gồm cả các lượt hỏi recall.

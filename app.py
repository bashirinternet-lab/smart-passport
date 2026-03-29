from flask import Flask, render_template, request, send_file
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageEnhance
import io

app = Flask(__name__)

# rembg এর সেশন আগে থেকেই লোড করে রাখছি যাতে ফাস্ট কাজ করে
session = new_session("u2net_human_seg")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image uploaded", 400
        
    file = request.files['image']
    copies = int(request.form.get('copies', 4))
    
    try:
        # ১. ছবি রিড করা
        input_image = Image.open(file.stream)
        
        # ২. ব্যাকগ্রাউন্ড রিমুভ করা
        output_image = remove(
            input_image, 
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        
        # ৩. সাদা ব্যাকগ্রাউন্ড বসানো
        bg_layer = Image.new("RGBA", output_image.size, "#FFFFFF")
        bg_layer.paste(output_image, (0, 0), output_image)
        final_img = bg_layer.convert("RGB")
        
        # ৪. একটু শার্পনেস বাড়িয়ে ছবি ক্লিয়ার করা (যাতে প্রিন্ট সুন্দর হয়)
        enhancer = ImageEnhance.Sharpness(final_img)
        final_img = enhancer.enhance(1.5)
        
        # ৫. পাসপোর্ট সাইজ ও বর্ডার (তোর ডেক্সটপ অ্যাপের মাপ অনুযায়ী)
        target_size = (450, 570)
        final_img = ImageOps.fit(final_img, target_size, Image.Resampling.LANCZOS)
        final_img = ImageOps.expand(final_img, border=3, fill='#808080')
        
        # ৬. A4 পেপার জেনারেট করা
        a4_w, a4_h = 2480, 3508
        a4_canvas = Image.new("RGB", (a4_w, a4_h), "white")
        
        img_w, img_h = final_img.size
        x, y = 40, 40
        gap = 50
        
        for _ in range(copies):
            a4_canvas.paste(final_img, (x, y))
            x += img_w + gap
            if x + img_w > a4_w:
                x = 40
                y += img_h + gap
                
        # ৭. প্রসেস করা ছবি মেমোরি থেকে সেন্ড করা
        img_io = io.BytesIO()
        a4_canvas.save(img_io, 'JPEG', quality=100)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)

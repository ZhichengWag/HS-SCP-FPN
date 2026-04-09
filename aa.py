import requests

url = "https://aistudio.baidu.com/llm/files/datasets/165532/file/1013058/download"

# 1. 填入你最新抓包获取的完整 Cookie（已修复末尾多余的引号）
cookies_str = "BAIDUID_BFESS=9DCDD460A72A63AC698DD6855EEB3A30:FG=1; __cas__st__533=NLI; __cas__id__533=0; __cas__rn__=0; jsdk-uuid=83344060-6c33-4613-b330-fc4ab7630e59; jsdk-uuid=83344060-6c33-4613-b330-fc4ab7630e59; Hm_lvt_be6b0f3e9ab579df8f47db4641a0a406=1775197259; HMACCOUNT=DC91FF838D6C36E9; ai-studio-lc=zh_CN; ppfuid=FOCoIC3q5fKa8fgJnwzbE67EJ49BGJeplOzf+4l4EOvDuu2RXBRv6R3A1AZMa49I27C0gDDLrJyxcIIeAeEhD8JYsoLTpBiaCXhLqvzbzmvy3SeAW17tKgNq/Xx+RgOdb8TWCFe62MVrDTY6lMf2GrfqL8c87KLF2qFER3obJGkLiHV2Ofz1ceQvHSB/JRIjGEimjy3MrXEpSuItnI4KDymT2lyQbips1ItIErRL+oahVhxaw/3fSGCUDnH8ryrFqqUTo6WPP4TTQmKEAyd5WBJsVwXkGdF24AsEQ3K5XBbh9EHAWDOg2T1ejpq0s2eFz67xpujr1V+of2ZgvZNzVPpaEhZpob9le2b5QIEdiQeMGGtnCq9fZ/aAmiQfaXFfXhLZ126CPFCIEuj/nWa+RPzg/CQV+t396RcQ+QB5B6TasmgOrJ40n63OsKSOpoSLpklEtTKfx5hHc9j4nurRraXUHgNWSPA31ou+XTSfsKyVXVSGGnkUB7qA0khSm2nsQwBpdgqbXUb4lU+zNAV2n0AktybhhKxhReRo8jZOXronbjpyaNZANNqEA4g1HmtdHmv/tVUjVExnyBvjSrtrPu8IrnpcuigpPlr6uwWt/lm7TLIKKJqASWGtMQ6010Ekmrx4fAQe4UGeL1qFLCkLuVsqRTBoofr21/QMVXuElE6IsRNIWWghWQd3Lf4jYlSvUuymUDPSEyRa3+0Ti1dVRXtBxMNNlZL/aYKhL8ZXc31rBDqIzqGz8FaOWCcRrX7mZxSsImZT5FjjbKQHRzCZQ+neA92A3SsiLF9YTT6DPLTkUprNdxn+PjQUm/6IsYGfeeSSAeWd7rUjwjsqG1CAideOw1L36a14YJP++P+kS+Sb5mK1x49t/bVAyV1GuW/Di4WdaYtTzhOHb0ypS5VHS/QoLlXJRRrnkhdcvkOKSMp9S91PAcsPMDdk9LSx3CI2+A4fy1sdGkfv+jOUK5r/COAFGQM9PC1eNi2DOZbvibR8NTYxlrGkFlkXxi6t+hxhyYN1zEUp6ggVCZWoCgBT7dhxdC3bEpOFlnJXW/ewK+fOJ1Rm0oz1xFFR9FYG1BJvnQA8z6Cz/jTzGDsocIHwA4qlml8ik+FDREmF7DwZHRpOpiwmjUhELAzdRtuu+0nt6o8w3MlwZxJhxBabUW5sicyie973hz6nxWLbBzvYx9F54WJPMynUbqkO3Z7jSA8MZt1Aj6NtrhSNGXID70JtNZcYUpaI4ZXCZpGIlSF+pB7KTLoU96dsmC3+9Ar1MCJM; BDUSS=3d4ekJWWW9BfjNNOVNHV2VxZnYyR2JpbzZwOXgtallZZGFSV0duSEgteFQ2dlpwRVFBQUFBJCQAAAAAAAAAAAEAAAAp22fx1NrPws6qzfUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFNdz2lTXc9pZ; BDUSS_BFESS=3d4ekJWWW9BfjNNOVNHV2VxZnYyR2JpbzZwOXgtallZZGFSV0duSEgteFQ2dlpwRVFBQUFBJCQAAAAAAAAAAAEAAAAp22fx1NrPws6qzfUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFNdz2lTXc9pZ; Hm_up_be6b0f3e9ab579df8f47db4641a0a406=%7B%22user_reg_date%22%3A%7B%22value%22%3A%2220220605%22%2C%22scope%22%3A1%7D%2C%22user_course_rt%22%3A%7B%22value%22%3A%22%E9%9D%9E%E8%AF%BE%E7%A8%8B%E7%94%A8%E6%88%B7%22%2C%22scope%22%3A1%7D%2C%22user_center_type%22%3A%7B%22value%22%3A%221%22%2C%22scope%22%3A1%7D%2C%22uid_%22%3A%7B%22value%22%3A%222385153%22%2C%22scope%22%3A1%7D%7D; ai-studio-ticket=C34A4C54A2BD4865AEC74D8A66D90B9DF21BCA7023204D8E9337E5C83DAD6606; Hm_lpvt_be6b0f3e9ab579df8f47db4641a0a406=1775731301; lang=zh; RT=z=1&dm=baidu.com&si=fecf9274-11c6-435c-aaf3-d72e141829c5&ss=mnr96k54&sl=n&tt=jqu&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&ld=3dyum&nu=9y8m6cy&cl=3rm0w"

# 2. 填入你最新抓包获取的 x-studio-token
x_token = "CE88240C2DA0E341A25AF8B9E043E19EADC7E7E90BF1F8BE6269253A0A9452A1E63B1383518F9DBEBCA2319E33DC67AC"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Referer": "https://aistudio.baidu.com/datasetdetail/165532",
    "Cookie": cookies_str,
    "x-studio-token": x_token  
}

print("开始连接并获取底层真实下载地址...")
response = requests.get(url, headers=headers)

try:
    res_json = response.json()
    # 判断是否成功获取到了数据
    if res_json.get('errorCode') == 0:
        file_url = res_json['result']['fileUrl']
        print("✅ 身份验证成功，拿到真实下载链接！")
        print("开始高速下载数据集文件 (AI-TOD.zip)...")
        
        # 第二步：直接请求那个真实的下载链接，不需要再带鉴权头了
        dl_resp = requests.get(file_url, stream=True)
        total_size = int(dl_resp.headers.get('content-length', 0))
        filename = "AI-TOD.zip"  # 根据你的 json，文件名叫 AI-TOD.zip
        
        with open(filename, 'wb') as file:
            downloaded = 0
            for data in dl_resp.iter_content(chunk_size=1024*1024):
                file.write(data)
                downloaded += len(data)
                if total_size > 0:
                    print(f"已下载: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB", end='\r')
                else:
                    print(f"已下载: {downloaded / (1024*1024):.2f} MB", end='\r')
        print("\n🎉 下载完成！文件已保存为 AI-TOD.zip")
        
    else:
        print("获取链接失败，可能是 Token 刚刚过期了：", res_json)
        
except Exception as e:
    print("发生意外错误，返回内容不是标准 JSON:", e)
    print("返回内容:", response.text[:500])
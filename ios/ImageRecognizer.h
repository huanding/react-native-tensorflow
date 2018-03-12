#import <Foundation/Foundation.h>

@interface ImageRecognizer: NSObject
- (id) initWithData:(NSString *)model labels:(NSString *)labels;
- (NSArray *) recognizeImage:(NSString *)image maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold;
@end

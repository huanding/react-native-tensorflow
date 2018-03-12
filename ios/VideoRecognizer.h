#import <Foundation/Foundation.h>

@interface VideoRecognizer: NSObject
- (id) initWithData:(NSString *)model labels:(NSString *)labels;
- (NSArray *) recognize:(NSString *)video maxResults:(NSNumber *)maxResults threshold:(NSNumber *)threshold;
@end
